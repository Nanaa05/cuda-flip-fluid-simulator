#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cub/cub.cuh>
#include <math.h>

extern void launchIntegrateParticles(DeviceData& d, int numParticles, float dt, float gravity);
extern void launchPushParticlesApart(DeviceData& d, int numParticles, int numIters, void* cub_temp, size_t cub_temp_bytes);
extern void launchHandleCollisions(DeviceData& d, int numParticles, float obstacleX, float obstacleY, float obstacleRadius, float obstacleVelX, float obstacleVelY);
extern void launchSavePrevVelocities(DeviceData& d, int fNumCells);
extern void launchClassifyCells(DeviceData& d, int fNumCells, int numParticles);
extern void launchP2G(DeviceData& d, int numParticles);
extern void launchRestoreSolidCells(DeviceData& d, int fNumCells);
extern void launchUpdateParticleDensity(DeviceData& d, int numParticles, int fNumCells);
extern float launchComputeRestDensity(DeviceData& d, int fNumCells);
extern void launchRedBlackSolver(DeviceData& d, int numIters, float dt, float overRelaxation, bool compensateDrift, float restDensity);
extern void launchG2P(DeviceData& d, int numParticles, float flipRatio);
extern void launchUpdateParticleColors(DeviceData& d, int numParticles);
extern void launchUpdateCellColors(DeviceData& d, int fNumCells);

// Static states for pipeline
static void* d_cub_temp = nullptr;
static size_t cub_temp_bytes = 0;
static float cachedRestDensity = 0.0f;
static bool isFirstFrame = true;

GpuTelemetry g_gpu_telemetry;
static cudaEvent_t ev_start = nullptr;
static cudaEvent_t ev_stop = nullptr;

// carveObstacle_kernel
// Optional kernel to rasterize moving obstacles directly onto the MAC grid solid flags
// - 1 thread per interior cell
// - if dist(cell_center, obstacle) < radius: s[i]=0, set u/v faces to obstacle vel
// - else: s[i]=1
__global__ void carveObstacle_kernel(
    float* s, float* u, float* v,
    float obsX, float obsY, float obsRadius,
    float obsVx, float obsVy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= d_params.fNumX - 1 || j < 1 || j >= d_params.fNumY - 1)
        return;

    int center = i * d_params.fNumY + j;

    // Cell center coordinates
    float cx = (i + 0.5f) * d_params.h;
    float cy = (j + 0.5f) * d_params.h;

    float dx = cx - obsX;
    float dy = cy - obsY;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < obsRadius) {
        s[center] = 0.0f; // Set as SOLID_CELL
        // Enforce obstacle boundary velocity on surrounding cell faces
        u[center] = obsVx;                      // Left face
        u[center + d_params.fNumY] = obsVx;     // Right face
        v[center] = obsVy;                      // Bottom face
        v[center + 1] = obsVy;                  // Top face
    } else {
        s[center] = 1.0f; // FLUID/AIR accessible
    }
}

// gpuSimulateInit
// - cudaMemcpyToSymbol(d_params, &params)
// - DeviceData::allocate()
// - cudaMemcpy initial posX/posY, s[] to device
// - zero vel, u, v, p, particleDensity on device
// - allocate CUB temp buffer for spatial hash scan (Node D)
// - call interopInit() if interop enabled
void gpuSimulateInit(DeviceData& d, SimParams& params, const float* hPosX,
                     const float* hPosY, const float* hS, int numParticles)
{
    // Upload parameter constants to device
    cudaMemcpyToSymbol(d_params, &params, sizeof(SimParams));

    // Allocate MAC Grid, Particles, and Hashes
    d.allocate(params.fNumCells, params.pNumCells, params.maxParticles);

    // Reset structural arrays, and copy over standard initial states
    d.reset(hPosX, hPosY, hS, numParticles, params.fNumCells);

    // Allocate CUB temporary buffer for spatial hash scanning (Node D)
    cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes, d.numCellParticles, d.firstCellParticle, params.pNumCells + 1);
    cudaMalloc(&d_cub_temp, cub_temp_bytes);

    isFirstFrame = true;
    cachedRestDensity = 0.0f;

    if (!ev_start) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
    }
}

// gpuSimulate
// pipeline per substep:
//   T1: launchIntegrateParticles
//   T2: launchPushParticlesApart  (if separateParticles)
//   T3: launchHandleCollisions
//   T4: launchSavePrevVelocities, launchClassifyCells, launchP2G, launchRestoreSolidCells
//   T5: launchUpdateParticleDensity
//       launchComputeRestDensity  (first frame only, sets particleRestDensity)
//   T6: launchRedBlackSolver
//   T7: launchG2P
// after substeps:
//   T8: launchComputeRestDensity, launchUpdateParticleColors, launchUpdateCellColors
void gpuSimulate(DeviceData& d, float dt, float gravity, float flipRatio,
                 int numPressureIters, int numParticleIters,
                 float overRelaxation, bool compensateDrift,
                 bool separateParticles, float obstacleX, float obstacleY,
                 float obstacleRadius, float obstacleVelX, float obstacleVelY,
                 int numSubSteps)
{
    SimParams params;
    cudaMemcpyFromSymbol(&params, d_params, sizeof(SimParams), 0, cudaMemcpyDeviceToHost);

    int numParticles = params.maxParticles;
    float subDt = dt / numSubSteps;

    g_gpu_telemetry.pressureIters = numPressureIters;
    float ms = 0.0f;
    double frame_t1 = 0.0, frame_t2 = 0.0, frame_t3 = 0.0;
    double frame_t4 = 0.0, frame_t5 = 0.0, frame_t6 = 0.0, frame_t7 = 0.0;

    // Optional: Carve moving obstacle dynamically into grid
    if (obstacleRadius > 0.0f) {
        dim3 threads(16, 16);
        dim3 blocks((params.fNumX + threads.x - 1) / threads.x,
                    (params.fNumY + threads.y - 1) / threads.y);

        carveObstacle_kernel<<<blocks, threads>>>(
            d.s, d.u, d.v,
            obstacleX, obstacleY, obstacleRadius,
            obstacleVelX, obstacleVelY
        );
    }

    // Main Integration Sub-steps
    for (int step = 0; step < numSubSteps; ++step) {
        // T1: Kinematics
        cudaEventRecord(ev_start);
        launchIntegrateParticles(d, numParticles, subDt, gravity);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t1 += ms;

        // T2: Spatial Hash & Separation
        cudaEventRecord(ev_start);
        if (separateParticles) {
            launchPushParticlesApart(d, numParticles, numParticleIters, d_cub_temp, cub_temp_bytes);
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t2 += ms;

        // T3: Dynamic Collisions
        cudaEventRecord(ev_start);
        launchHandleCollisions(d, numParticles, obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t3 += ms;

        // T4: P2G Transfer & Boundary Recovery
        cudaEventRecord(ev_start);
        launchSavePrevVelocities(d, params.fNumCells);
        launchClassifyCells(d, params.fNumCells, numParticles);
        launchP2G(d, numParticles);
        launchRestoreSolidCells(d, params.fNumCells);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t4 += ms;

        // T5: Volume Calculation (Density Field)
        cudaEventRecord(ev_start);
        launchUpdateParticleDensity(d, numParticles, params.fNumCells);
        if (isFirstFrame) {
            // Establishes baseline rest density during the very first substep
            cachedRestDensity = launchComputeRestDensity(d, params.fNumCells);
            isFirstFrame = false;
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t5 += ms;

        // T6: Red-Black Gauss-Seidel Solver (Divergence / Pressure)
        cudaEventRecord(ev_start);
        launchRedBlackSolver(d, numPressureIters, subDt, overRelaxation, compensateDrift, cachedRestDensity);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t6 += ms;

        // T7: G2P Transfer (PIC / FLIP combination)
        cudaEventRecord(ev_start);
        launchG2P(d, numParticles, flipRatio);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t7 += ms;
    }

    g_gpu_telemetry.t1 += frame_t1;
    g_gpu_telemetry.t2 += frame_t2;
    g_gpu_telemetry.t3 += frame_t3;
    g_gpu_telemetry.t4 += frame_t4;
    g_gpu_telemetry.t5 += frame_t5;
    g_gpu_telemetry.t6 += frame_t6;
    g_gpu_telemetry.t7 += frame_t7;

    // T8: Post-Substep Reductions and Rendering Metadata
    cudaEventRecord(ev_start);
    cachedRestDensity = launchComputeRestDensity(d, params.fNumCells);
    launchUpdateParticleColors(d, numParticles);
    launchUpdateCellColors(d, params.fNumCells);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    g_gpu_telemetry.t8 += ms;
}
