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

    float subDt = dt / numSubSteps;

    if (obstacleRadius > 0.0f) {
        dim3 threads(16, 16);
        dim3 blocks((params.fNumX + 15) / 16, (params.fNumY + 15) / 16);
        carveObstacle_kernel<<<blocks, threads>>>(
            d.s, d.u, d.v,
            obstacleX, obstacleY, obstacleRadius,
            obstacleVelX, obstacleVelY
        );
    }

    for (int step = 0; step < numSubSteps; ++step) {
        // T1: integrate particles
        launchIntegrateParticles(d, numParticles, subDt, gravity);

        // T2: push particles apart
        if (separateParticles)
            launchPushParticlesApart(d, numParticles, numParticleIters, d_cub_temp, cub_temp_bytes);

        // T3: handle collisions
        launchHandleCollisions(d, numParticles, obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY);

        // T4: P2G transfer
        launchSavePrevVelocities(d, params.fNumCells);
        launchClassifyCells(d, params.fNumCells, numParticles, params.fInvSpacing, params.fNumX, params.fNumY);
        launchP2G(d, numParticles, params.fNumCells, params.h, params.fInvSpacing, params.fNumX, params.fNumY);
        launchRestoreSolidCells(d, params.fNumX, params.fNumY);

        // T5: density field
        launchUpdateParticleDensity(d, numParticles, params.fNumCells, params.h, params.fInvSpacing, params.fNumX, params.fNumY);
        if (isFirstFrame) {
            cachedRestDensity = launchComputeRestDensity(d, params.fNumCells);
            isFirstFrame = false;
        }

        // T6: pressure solve
        launchRedBlackSolver(d, numPressureIters, subDt, overRelaxation, compensateDrift, cachedRestDensity);

        // T7: G2P transfer
        launchG2P(d, numParticles, flipRatio, params.h, params.fInvSpacing, params.fNumX, params.fNumY);
    }

}

// === gpuUpdateColors: T8 recompute rest density and update particle/cell colors ===
void gpuUpdateColors(DeviceData& d, int numParticles) {
    SimParams params;
    cudaMemcpyFromSymbol(&params, d_params, sizeof(SimParams), 0, cudaMemcpyDeviceToHost);
    cachedRestDensity = launchComputeRestDensity(d, params.fNumCells);
    launchUpdateParticleColors(d, numParticles, cachedRestDensity);
    launchUpdateCellColors(d, params.fNumCells, cachedRestDensity);
}
