#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <device_launch_parameters.h>

// pressureSolve_kernel(int parity)
// - parity 0 = RED cells (i+j) % 2 == 0
// - parity 1 = BLACK cells (i+j) % 2 == 1
// - 1 thread per cell, 2D grid (fNumX x fNumY)
// - skip if wrong parity, not FLUID_CELL, or out of bounds
// - div = u[right] - u[center] + v[top] - v[center]
// - if compensateDrift: div -= k * max(0, particleDensity[center] - restDensity)
// - pVal = -(div / sSum) * overRelaxation
// - update p[center], u/v faces of center and 4 neighbors
// - called alternately. RED pass, BLACK pass, for numIters/2 rounds

__global__ void pressureSolve_kernel(
    int parity,
    float* u, float* v, float* p, const float* s,
    const int* cellType, const float* particleDensity,
    float overRelaxation, bool compensateDrift, float restDensity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    // skip the outermost layers
    if (i < 1 || i >= d_params.fNumX - 1 || j < 1 || j >= d_params.fNumY - 1)
        return;

    if ((i + j) % 2 != parity)
        return;

    // Linear index for standard fNumX x fNumY row-major mapping (column-major based on your stride)
    int center = i * d_params.fNumY + j;

    if (cellType[center] != FLUID_CELL)
        return;

    // MAC grid neighbor indices
    int left   = (i - 1) * d_params.fNumY + j;
    int right  = (i + 1) * d_params.fNumY + j;
    int bottom = i * d_params.fNumY + (j - 1);
    int top    = i * d_params.fNumY + (j + 1);

    float sLeft   = s[left];
    float sRight  = s[right];
    float sBottom = s[bottom];
    float sTop    = s[top];

    float sSum = sLeft + sRight + sBottom + sTop;
    if (sSum == 0.0f) return; // solid surrounds

    // velocity divergence
    // u=left face, v=bottom face on MAC grid
    float div = u[right] - u[center] + v[top] - v[center];

    // volume preservation/drift compensation
    if (compensateDrift) {
        // drift coefficient
        // 1.0f acts as a solid base modifier for typical FLIP setups
        const float k_drift = 1.0f;
        float densityErr = particleDensity[center] - restDensity;
        if (densityErr > 0.0f) {
            div -= k_drift * densityErr;
        }
    }

    float pVal = -(div / sSum) * overRelaxation;

    // update center pressure
    p[center] += pVal;

    // Apply pressure gradient to velocities (scatter)
    // Left face (u[center]) gets pushed left, making velocity more negative
    // Right face (u[right]) gets pushed right, making velocity more positive
    u[center] -= sLeft   * pVal;
    u[right]  += sRight  * pVal;
    v[center] -= sBottom * pVal;
    v[top]    += sTop    * pVal;
}

void launchRedBlackSolver(DeviceData& d, int numIters, float dt,
                          float overRelaxation, bool compensateDrift,
                          float restDensity)
{
    // Read MAC grid dimensions from the device constant memory symbol `d_params`
    int fNumX, fNumY;
    cudaMemcpyFromSymbol(&fNumX, d_params, sizeof(int), offsetof(SimParams, fNumX), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&fNumY, d_params, sizeof(int), offsetof(SimParams, fNumY), cudaMemcpyDeviceToHost);

    // 2D thread block
    dim3 threads(16, 16);
    dim3 blocks((fNumX + threads.x - 1) / threads.x,
                (fNumY + threads.y - 1) / threads.y);

    // Run the solver. 1 iteration = 1 full pass of RED + BLACK cells.
    for (int iter = 0; iter < numIters; ++iter) {

        // RED pass (parity = 0)
        pressureSolve_kernel<<<blocks, threads>>>(
            0, d.u, d.v, d.p, d.s,
            d.cellType, d.particleDensity,
            overRelaxation, compensateDrift, restDensity
        );

        // BLACK pass (parity = 1)
        pressureSolve_kernel<<<blocks, threads>>>(
            1, d.u, d.v, d.p, d.s,
            d.cellType, d.particleDensity,
            overRelaxation, compensateDrift, restDensity
        );
    }
}
