#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cub/cub.cuh>
#include <math.h>

extern void launchIntegrateParticles(DeviceData& d, int numParticles, float dt, float gravity);
extern void launchPushParticlesApart(DeviceData& d, int numParticles, int numIters, void* cub_temp, size_t cub_temp_bytes);
extern void launchHandleCollisions(DeviceData& d, int numParticles, float obstacleX, float obstacleY, float obstacleRadius, float obstacleVelX, float obstacleVelY);
extern void launchSavePrevVelocities(DeviceData& d, int fNumCells);
extern void launchClassifyCells(DeviceData& d, int fNumCells, int numParticles, float fInvSpacing, int fNumX, int fNumY);
extern void launchP2G(DeviceData& d, int numParticles, int fNumCells, float h, float fInvSpacing, int fNumX, int fNumY);
extern void launchRestoreSolidCells(DeviceData& d, int fNumX, int fNumY);
extern void launchUpdateParticleDensity(DeviceData& d, int numParticles, int fNumCells, float h, float fInvSpacing, int fNumX, int fNumY);
extern float launchComputeRestDensity(DeviceData& d, int fNumCells);
extern void launchRedBlackSolver(DeviceData& d, int numIters, float dt, float overRelaxation, bool compensateDrift, float restDensity);
extern void launchG2P(DeviceData& d, int numParticles, float flipRatio, float h, float fInvSpacing, int fNumX, int fNumY);
extern void launchUpdateParticleColors(DeviceData& d, int numParticles, float restDensity);
extern void launchUpdateCellColors(DeviceData& d, int fNumCells, float restDensity);

static void* d_cub_temp = nullptr;
static size_t cub_temp_bytes = 0;
static float cachedRestDensity = 0.0f;
static bool isFirstFrame = true;

GpuTelemetry g_gpu_telemetry;
static cudaEvent_t ev_start = nullptr;
static cudaEvent_t ev_stop = nullptr;

// === carveObstacle_kernel: rasterize moving obstacle into s[], u[], v[] ===
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
    float cx = (i + 0.5f) * d_params.h;
    float cy = (j + 0.5f) * d_params.h;
    float dx = cx - obsX;
    float dy = cy - obsY;

    if (dx * dx + dy * dy < obsRadius * obsRadius) {
        s[center] = 0.0f;
        u[center] = obsVx;
        u[center + d_params.fNumY] = obsVx;
        v[center] = obsVy;
        v[center + 1] = obsVy;
    } else {
        s[center] = 1.0f;
    }
}

// === gpuSimulate: full FLIP pipeline per substep ===
void gpuSimulate(DeviceData& d, int numParticles, float dt, float gravity, float flipRatio,
                 int numPressureIters, int numParticleIters,
                 float overRelaxation, bool compensateDrift,
                 bool separateParticles, float obstacleX, float obstacleY,
                 float obstacleRadius, float obstacleVelX, float obstacleVelY,
                 int numSubSteps)
{
    SimParams params;
    cudaMemcpyFromSymbol(&params, d_params, sizeof(SimParams), 0, cudaMemcpyDeviceToHost);

    if (separateParticles && d_cub_temp == nullptr) {
        cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes,
            d.numCellParticles, d.firstCellParticle, params.pNumCells + 1);
        cudaMalloc(&d_cub_temp, cub_temp_bytes);
    }

    if (ev_start == nullptr) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
    }

    float subDt = dt / numSubSteps;

    g_gpu_telemetry.pressureIters = numPressureIters;
    float ms = 0.0f;
    double frame_t1 = 0.0, frame_t2 = 0.0, frame_t3 = 0.0;
    double frame_t4 = 0.0, frame_t5 = 0.0, frame_t6 = 0.0, frame_t7 = 0.0;

    if (obstacleRadius > 0.0f) {
        dim3 threads(16, 16);
        dim3 blocks((params.fNumX + 15) / 16, (params.fNumY + 15) / 16);
        
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
        launchClassifyCells(d, params.fNumCells, numParticles, params.fInvSpacing, params.fNumX, params.fNumY);
        launchP2G(d, numParticles, params.fNumCells, params.h, params.fInvSpacing, params.fNumX, params.fNumY);
        launchRestoreSolidCells(d, params.fNumX, params.fNumY);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t4 += ms;

        // T5: Volume Calculation (Density Field)
        cudaEventRecord(ev_start);
        launchUpdateParticleDensity(d, numParticles, params.fNumCells, params.h, params.fInvSpacing, params.fNumX, params.fNumY);
        if (isFirstFrame) {
            cachedRestDensity = launchComputeRestDensity(d, params.fNumCells);
            isFirstFrame = false;
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t5 += ms;

        // T6: Red-Black Gauss-Seidel Solver (Divergence / Pressure)
        cudaEventRecord(ev_start);
        cudaMemcpy(d.prevU, d.u, params.fNumCells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d.prevV, d.v, params.fNumCells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(d.p, 0, params.fNumCells * sizeof(float));
        launchRedBlackSolver(d, numPressureIters, subDt, overRelaxation, compensateDrift, cachedRestDensity);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t6 += ms;

        // T7: G2P Transfer (PIC / FLIP combination)
        cudaEventRecord(ev_start);
        launchG2P(d, numParticles, flipRatio, params.h, params.fInvSpacing, params.fNumX, params.fNumY);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        frame_t7 += ms;
    }

    // Dorong akumulasi sub-step ke struktur global pencatatan
    g_gpu_telemetry.t1 += frame_t1;
    g_gpu_telemetry.t2 += frame_t2;
    g_gpu_telemetry.t3 += frame_t3;
    g_gpu_telemetry.t4 += frame_t4;
    g_gpu_telemetry.t5 += frame_t5;
    g_gpu_telemetry.t6 += frame_t6;
    g_gpu_telemetry.t7 += frame_t7;

    // T8: Post-Substep Reductions and Rendering Metadata
    cudaEventRecord(ev_start);
    launchUpdateParticleColors(d, numParticles, cachedRestDensity);
    launchUpdateCellColors(d, params.fNumCells, cachedRestDensity);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    g_gpu_telemetry.t8 += ms;
}

// === gpuUpdateColors: T8 update particle/cell colors using frozen rest density ===
void gpuUpdateColors(DeviceData& d, int numParticles) {
    SimParams params;
    cudaMemcpyFromSymbol(&params, d_params, sizeof(SimParams), 0, cudaMemcpyDeviceToHost);
    launchUpdateParticleColors(d, numParticles, cachedRestDensity);
    launchUpdateCellColors(d, params.fNumCells, cachedRestDensity);
}
