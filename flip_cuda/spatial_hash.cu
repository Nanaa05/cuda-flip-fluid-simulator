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

__global__ void histogram_kernel(
    float* posX, float* posY, int* numCellParticles, int numParticles) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        int xi = clampi_d(__float2int_rd(posX[i] * d_params.pInvSpacing), 0, d_params.pNumX - 1);
        int yi = clampi_d(__float2int_rd(posY[i] * d_params.pInvSpacing), 0, d_params.pNumY - 1);
        int cellNr = xi * d_params.pNumY + yi;
        
        atomicAdd(&numCellParticles[cellNr], 1);
    }
}

__global__ void scatter_kernel(
    float* posX, float* posY, 
    int* firstCellParticle_copy, int* cellParticleIds, 
    int numParticles) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        int xi = clampi_d(__float2int_rd(posX[i] * d_params.pInvSpacing), 0, d_params.pNumX - 1);
        int yi = clampi_d(__float2int_rd(posY[i] * d_params.pInvSpacing), 0, d_params.pNumY - 1);
        int cellNr = xi * d_params.pNumY + yi;
        

        int slot = atomicAdd(&firstCellParticle_copy[cellNr], 1);
        cellParticleIds[slot] = i;
    }
}

__global__ void separate_kernel(
    float* posX, float* posY, 
    float* colorR, float* colorG, float* colorB,
    int* firstCellParticle, int* cellParticleIds, 
    int numParticles) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float px = posX[i];
    float py = posY[i];

    int pxi = __float2int_rd(px * d_params.pInvSpacing);
    int pyi = __float2int_rd(py * d_params.pInvSpacing);
    
    int x0 = max(pxi - 1, 0);
    int y0 = max(pyi - 1, 0);
    int x1 = min(pxi + 1, d_params.pNumX - 1);
    int y1 = min(pyi + 1, d_params.pNumY - 1);

    float minDist = 2.0f * d_params.particleRadius;
    float minDist2 = minDist * minDist;
    const float colorDiffusionCoeff = 0.001f;

    float dx_accum = 0.0f;
    float dy_accum = 0.0f;

    for (int xi = x0; xi <= x1; ++xi) {
        for (int yi = y0; yi <= y1; ++yi) {
            int cellNr = xi * d_params.pNumY + yi;
            
            int firstI = firstCellParticle[cellNr];
            int lastI  = firstCellParticle[cellNr + 1];
            
            for (int j = firstI; j < lastI; ++j) {
                int idn = cellParticleIds[j];
                if (idn == i) continue;
                
                float qx = posX[idn];
                float qy = posY[idn];

                float dx = qx - (px - dx_accum); 
                float dy = qy - (py - dy_accum);
                float d2 = dx * dx + dy * dy;

                if (d2 > minDist2 || d2 == 0.0f) continue;
                
                float d = sqrtf(d2);
                float sFac = 0.5f * (minDist - d) / d;
                dx *= sFac;
                dy *= sFac;

                dx_accum += dx;
                dy_accum += dy;

                atomicAdd(&posX[idn], dx);
                atomicAdd(&posY[idn], dy);

                float c0r = colorR[i], c1r = colorR[idn];
                float c0g = colorG[i], c1g = colorG[idn];
                float c0b = colorB[i], c1b = colorB[idn];
                
                float cr = (c0r + c1r) * 0.5f;
                float cg = (c0g + c1g) * 0.5f;
                float cb = (c0b + c1b) * 0.5f;

                colorR[i]   = c0r + (cr - c0r) * colorDiffusionCoeff;
                colorR[idn] = c1r + (cr - c1r) * colorDiffusionCoeff;
                colorG[i]   = c0g + (cg - c0g) * colorDiffusionCoeff;
                colorG[idn] = c1g + (cg - c1g) * colorDiffusionCoeff;
                colorB[i]   = c0b + (cb - c0b) * colorDiffusionCoeff;
                colorB[idn] = c1b + (cb - c1b) * colorDiffusionCoeff;
            }
        }
    }

    if (dx_accum != 0.0f || dy_accum != 0.0f) {
        atomicAdd(&posX[i], -dx_accum);
        atomicAdd(&posY[i], -dy_accum);
    }
}

void launchPushParticlesApart(DeviceData& d, int numParticles, int numIters, void* cub_temp, size_t cub_temp_bytes) {
    if (numParticles <= 0) return;

    int* firstCellParticle_copy = (int*)d.d_partialSums;

    int pNumCells;
    cudaMemcpyFromSymbol(&pNumCells, d_params, sizeof(int), offsetof(SimParams, pNumCells), cudaMemcpyDeviceToHost);

    int threadsPerBlock = 256;
    int particleBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(d.numCellParticles, 0, pNumCells * sizeof(int));

    histogram_kernel<<<particleBlocks, threadsPerBlock>>>(
        d.posX, d.posY, d.numCellParticles, numParticles
    );

    cub::DeviceScan::ExclusiveSum(
        cub_temp, cub_temp_bytes, 
        d.numCellParticles, d.firstCellParticle, pNumCells + 1
    );

    cudaMemcpy(firstCellParticle_copy, d.firstCellParticle, (pNumCells + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    scatter_kernel<<<particleBlocks, threadsPerBlock>>>(
        d.posX, d.posY, 
        firstCellParticle_copy, d.cellParticleIds, 
        numParticles
    );

    for (int iter = 0; iter < numIters; ++iter) {
        separate_kernel<<<particleBlocks, threadsPerBlock>>>(
            d.posX, d.posY, 
            d.colorR, d.colorG, d.colorB,
            d.firstCellParticle, d.cellParticleIds, 
            numParticles
        );
    }
}
