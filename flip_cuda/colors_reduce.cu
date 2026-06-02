#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector>

// === setSciColor_d: map density value to jet colormap, write to cellColor ===
__device__ void setSciColor_d(float* cellColor, int cellNr, float val, float minVal, float maxVal) {
    float v = fmaxf(minVal, fminf(maxVal, val));
    float dv = maxVal - minVal;
    float r = 1.0f, g = 1.0f, b = 1.0f;
    if (dv > 0.0f) v = (v - minVal) / dv;
    else v = 0.5f;
    if (v < 0.25f) {
        r = 0.0f;
        g = 4.0f * v;
    } else if (v < 0.5f) {
        r = 0.0f;
        b = 1.0f - 4.0f * (v - 0.25f);
    } else if (v < 0.75f) {
        r = 4.0f * (v - 0.5f);
        b = 0.0f;
    } else {
        g = 1.0f - 4.0f * (v - 0.75f);
        b = 0.0f;
    }
    cellColor[3 * cellNr + 0] = r;
    cellColor[3 * cellNr + 1] = g;
    cellColor[3 * cellNr + 2] = b;
}

// === computeRestDensity_kernel: parallel reduction over fluid cells ===
__global__ void computeRestDensity_kernel(
    const float* particleDensity, const int* cellType,
    float* d_partialSums, int* d_fluidCount, int fNumCells)
{
    extern __shared__ float s_data[];
    __shared__ int s_count[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    int count = 0;
    if (i < fNumCells && cellType[i] == FLUID_CELL) {
        sum = particleDensity[i];
        count = 1;
    }
    s_data[tid] = sum;
    s_count[tid] = count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_partialSums[blockIdx.x] = s_data[0];
        atomicAdd(d_fluidCount, s_count[0]);
    }
}

// === updateParticleColors_kernel: drift colors, mark low-density particles white-blue ===
__global__ void updateParticleColors_kernel(
    float* colorR, float* colorG, float* colorB,
    const float* posX, const float* posY, const float* particleDensity,
    float restDensity, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int xi = max(0, min((int)(posX[i] * d_params.fInvSpacing), d_params.fNumX - 1));
    int yi = max(0, min((int)(posY[i] * d_params.fInvSpacing), d_params.fNumY - 1));
    int cellNr = xi * d_params.fNumY + yi;

    float densityRatio = (restDensity > 0.0f) ? (particleDensity[cellNr] / restDensity) : 1.0f;

    if (densityRatio < 0.7f) {
        colorR[i] = 0.8f;
        colorG[i] = 0.8f;
        colorB[i] = 1.0f;
    } else {
        colorR[i] = fmaxf(0.0f, fminf(1.0f, colorR[i] - 0.01f));
        colorG[i] = fmaxf(0.0f, fminf(1.0f, colorG[i] - 0.01f));
        colorB[i] = fmaxf(0.0f, fminf(1.0f, colorB[i] + 0.01f));
    }
}

// === updateCellColors_kernel: solid=gray, fluid=jet(density), air=black ===
__global__ void updateCellColors_kernel(
    float* cellColor, const int* cellType,
    const float* particleDensity, float restDensity, int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;

    int type = cellType[i];
    if (type == SOLID_CELL) {
        cellColor[3 * i + 0] = 0.5f;
        cellColor[3 * i + 1] = 0.5f;
        cellColor[3 * i + 2] = 0.5f;
    } else if (type == FLUID_CELL) {
        setSciColor_d(cellColor, i, particleDensity[i], 0.0f, 2.0f * restDensity);
    } else {
        cellColor[3 * i + 0] = 0.0f;
        cellColor[3 * i + 1] = 0.0f;
        cellColor[3 * i + 2] = 0.0f;
    }
}

// === launchComputeRestDensity: block-level reduction, returns average fluid density ===
float launchComputeRestDensity(DeviceData& d, int fNumCells) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (fNumCells + threadsPerBlock - 1) / threadsPerBlock;

    int* d_fluidCount = nullptr;
    cudaMalloc(&d_fluidCount, sizeof(int));
    cudaMemset(d_fluidCount, 0, sizeof(int));

    computeRestDensity_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d.particleDensity, d.cellType, d.d_partialSums, d_fluidCount, fNumCells
    );

    std::vector<float> h_partialSums(blocksPerGrid);
    cudaMemcpy(h_partialSums.data(), d.d_partialSums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    int h_fluidCount = 0;
    cudaMemcpy(&h_fluidCount, d_fluidCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_fluidCount);

    float total = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) total += h_partialSums[i];

    return (h_fluidCount > 0) ? (total / (float)h_fluidCount) : 0.0f;
}

// === launchUpdateParticleColors ===
void launchUpdateParticleColors(DeviceData& d, int numParticles, float restDensity) {
    if (numParticles <= 0) return;
    int blocks = (numParticles + 255) / 256;
    updateParticleColors_kernel<<<blocks, 256>>>(
        d.colorR, d.colorG, d.colorB,
        d.posX, d.posY, d.particleDensity,
        restDensity, numParticles
    );
}

// === launchUpdateCellColors ===
void launchUpdateCellColors(DeviceData& d, int fNumCells, float restDensity) {
    if (fNumCells <= 0) return;
    int blocks = (fNumCells + 255) / 256;
    updateCellColors_kernel<<<blocks, 256>>>(
        d.cellColor, d.cellType, d.particleDensity, restDensity, fNumCells
    );
}
