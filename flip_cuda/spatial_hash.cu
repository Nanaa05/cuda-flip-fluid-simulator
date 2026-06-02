#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cub/cub.cuh>
#include <device_launch_parameters.h>

__device__ inline float clampf_d(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

__device__ inline int clampi_d(int x, int lo, int hi) {
    return max(lo, min(hi, x));
}

// === histogram_kernel: count particles per spatial hash cell ===
__global__ void histogram_kernel(
    float* posX, float* posY, int* numCellParticles, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi = clampi_d(__float2int_rd(posX[i] * d_params.pInvSpacing), 0, d_params.pNumX - 1);
    int yi = clampi_d(__float2int_rd(posY[i] * d_params.pInvSpacing), 0, d_params.pNumY - 1);
    atomicAdd(&numCellParticles[xi * d_params.pNumY + yi], 1);
}

// === scatter_kernel: assign particle IDs into sorted cell slots ===
__global__ void scatter_kernel(
    float* posX, float* posY,
    int* firstCellParticle_copy, int* cellParticleIds,
    int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi = clampi_d(__float2int_rd(posX[i] * d_params.pInvSpacing), 0, d_params.pNumX - 1);
    int yi = clampi_d(__float2int_rd(posY[i] * d_params.pInvSpacing), 0, d_params.pNumY - 1);
    int cellNr = xi * d_params.pNumY + yi;
    int slot = atomicAdd(&firstCellParticle_copy[cellNr], 1);
    cellParticleIds[slot] = i;
}

// === separate_kernel: push overlapping particles apart, diffuse colors ===
__global__ void separate_kernel(
    const float* posInX, const float* posInY,
    float* posOutX, float* posOutY,
    float* colorR, float* colorG, float* colorB,
    int* firstCellParticle, int* cellParticleIds,
    int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float px = posInX[i];
    float py = posInY[i];

    int pxi = __float2int_rd(px * d_params.pInvSpacing);
    int pyi = __float2int_rd(py * d_params.pInvSpacing);

    int x0 = max(pxi - 1, 0);
    int y0 = max(pyi - 1, 0);
    int x1 = min(pxi + 1, d_params.pNumX - 1);
    int y1 = min(pyi + 1, d_params.pNumY - 1);

    float minDist = 2.0f * d_params.particleRadius;
    float minDist2 = minDist * minDist;
    const float colorDiffusion = 0.001f;
    const float relax = 0.5f;

    float dispX = 0.0f;
    float dispY = 0.0f;
    float cr = colorR[i];
    float cg = colorG[i];
    float cb = colorB[i];

    for (int xi = x0; xi <= x1; ++xi) {
        for (int yi = y0; yi <= y1; ++yi) {
            int cellNr = xi * d_params.pNumY + yi;
            int firstI = firstCellParticle[cellNr];
            int lastI = firstCellParticle[cellNr + 1];
            for (int j = firstI; j < lastI; ++j) {
                int idn = cellParticleIds[j];
                if (idn == i) continue;
                float qx = posInX[idn];
                float qy = posInY[idn];
                float dx = px - qx;
                float dy = py - qy;
                float d2 = dx * dx + dy * dy;
                if (d2 > minDist2 || d2 == 0.0f) continue;
                float dist = sqrtf(d2);
                float sFac = 0.5f * (minDist - dist) / dist;
                dispX += dx * sFac;
                dispY += dy * sFac;
                float mr = (cr + colorR[idn]) * 0.5f;
                float mg = (cg + colorG[idn]) * 0.5f;
                float mb = (cb + colorB[idn]) * 0.5f;
                cr += (mr - cr) * colorDiffusion;
                cg += (mg - cg) * colorDiffusion;
                cb += (mb - cb) * colorDiffusion;
            }
        }
    }

    posOutX[i] = px + dispX * relax;
    posOutY[i] = py + dispY * relax;
    colorR[i] = cr;
    colorG[i] = cg;
    colorB[i] = cb;
}

// === launchPushParticlesApart: rebuild spatial hash each iter, run separate_kernel ===
void launchPushParticlesApart(DeviceData& d, int numParticles, int numIters, void* cub_temp, size_t cub_temp_bytes) {
    if (numParticles <= 0) return;

    int* firstCellParticle_copy = (int*)d.d_partialSums;

    int pNumCells;
    cudaMemcpyFromSymbol(&pNumCells, d_params, sizeof(int), offsetof(SimParams, pNumCells), cudaMemcpyDeviceToHost);

    int blocks = (numParticles + 255) / 256;

    cudaMemset(d.numCellParticles, 0, pNumCells * sizeof(int));

    histogram_kernel<<<blocks, 256>>>(d.posX, d.posY, d.numCellParticles, numParticles);

    cub::DeviceScan::ExclusiveSum(
        cub_temp, cub_temp_bytes,
        d.numCellParticles, d.firstCellParticle, pNumCells + 1
    );

    cudaMemcpy(firstCellParticle_copy, d.firstCellParticle, (pNumCells + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    scatter_kernel<<<blocks, 256>>>(
        d.posX, d.posY,
        firstCellParticle_copy, d.cellParticleIds,
        numParticles
    );

    for (int iter = 0; iter < numIters; ++iter) {
        cudaMemcpy(d.sepPosX, d.posX, numParticles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d.sepPosY, d.posY, numParticles * sizeof(float), cudaMemcpyDeviceToDevice);
        separate_kernel<<<blocks, 256>>>(
            d.sepPosX, d.sepPosY,
            d.posX, d.posY,
            d.colorR, d.colorG, d.colorB,
            d.firstCellParticle, d.cellParticleIds,
            numParticles
        );
    }
}
