#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <device_launch_parameters.h>

// === pressureSolve_kernel: red-black SOR, parity 0=RED parity 1=BLACK ===
__global__ void pressureSolve_kernel(
    int parity,
    float* u, float* v, float* p, const float* s,
    const int* cellType, const float* particleDensity,
    float overRelaxation, bool compensateDrift, float restDensity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= d_params.fNumX - 1 || j < 1 || j >= d_params.fNumY - 1)
        return;
    if ((i + j) % 2 != parity)
        return;

    int center = i * d_params.fNumY + j;
    if (cellType[center] != FLUID_CELL)
        return;

    int left = (i - 1) * d_params.fNumY + j;
    int right = (i + 1) * d_params.fNumY + j;
    int bottom = i * d_params.fNumY + (j - 1);
    int top = i * d_params.fNumY + (j + 1);

    float sLeft = s[left];
    float sRight = s[right];
    float sBottom = s[bottom];
    float sTop = s[top];

    float sSum = sLeft + sRight + sBottom + sTop;
    if (sSum == 0.0f) return;

    float div = u[right] - u[center] + v[top] - v[center];

    if (compensateDrift) {
        const float k_drift = 1.0f;
        float densityErr = particleDensity[center] - restDensity;
        if (densityErr > 0.0f)
            div -= k_drift * densityErr;
    }

    float pVal = -(div / sSum) * overRelaxation;

    p[center] += pVal;
    u[center] -= sLeft * pVal;
    u[right] += sRight * pVal;
    v[center] -= sBottom * pVal;
    v[top] += sTop * pVal;
}

// === launchRedBlackSolver: numIters full red+black passes ===
void launchRedBlackSolver(DeviceData& d, int numIters, float dt,
                          float overRelaxation, bool compensateDrift,
                          float restDensity)
{
    int fNumX, fNumY;
    cudaMemcpyFromSymbol(&fNumX, d_params, sizeof(int), offsetof(SimParams, fNumX), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&fNumY, d_params, sizeof(int), offsetof(SimParams, fNumY), cudaMemcpyDeviceToHost);

    dim3 threads(16, 16);
    dim3 blocks((fNumX + threads.x - 1) / threads.x,
                (fNumY + threads.y - 1) / threads.y);

    for (int iter = 0; iter < numIters; ++iter) {
        pressureSolve_kernel<<<blocks, threads>>>(
            0, d.u, d.v, d.p, d.s,
            d.cellType, d.particleDensity,
            overRelaxation, compensateDrift, restDensity
        );
        pressureSolve_kernel<<<blocks, threads>>>(
            1, d.u, d.v, d.p, d.s,
            d.cellType, d.particleDensity,
            overRelaxation, compensateDrift, restDensity
        );
    }
}
