#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cuda_runtime.h>

// === clampf_d ===
static __device__ __forceinline__ float clampf_d(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

// === savePrevVelocities_kernel ===
__global__ void savePrevVelocities_kernel(
    float* u, float* v,
    float* du, float* dv,
    float* prevU, float* prevV,
    int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    prevU[i] = u[i];
    prevV[i] = v[i];
    du[i] = 0.0f;
    dv[i] = 0.0f;
    u[i] = 0.0f;
    v[i] = 0.0f;
}

// === classifyCellsA_kernel ===
__global__ void classifyCellsA_kernel(int* cellType, const float* s, int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    cellType[i] = (s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
}

// === classifyCellsB_kernel ===
__global__ void classifyCellsB_kernel(
    int* cellType,
    const float* posX, const float* posY,
    int numParticles,
    float fInvSpacing, int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi = max(0, min((int)floorf(posX[i] * fInvSpacing), fNumX - 1));
    int yi = max(0, min((int)floorf(posY[i] * fInvSpacing), fNumY - 1));
    atomicCAS(&cellType[xi * fNumY + yi], AIR_CELL, FLUID_CELL);
}

// === p2gScatter_kernel ===
__global__ void p2gScatter_kernel(
    float* fld, float* fldD,
    const float* posX, const float* posY,
    const float* particleVel,
    int numParticles, int component,
    float h, float fInvSpacing,
    int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int n = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * hh;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2 : 0.0f;

    float x = clampf_d(posX[i], hh, (fNumX - 1) * hh);
    float y = clampf_d(posY[i], hh, (fNumY - 1) * hh);

    int x0 = min((int)floorf((x - dxOff) * h1), fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int x1 = min(x0 + 1, fNumX - 2);

    int y0 = min((int)floorf((y - dyOff) * h1), fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx;
    float sy = 1.0f - ty;
    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;

    int nr0 = x0 * n + y0;
    int nr1 = x1 * n + y0;
    int nr2 = x1 * n + y1;
    int nr3 = x0 * n + y1;

    float pv = particleVel[i];
    atomicAdd(&fld[nr0], pv * d0);
    atomicAdd(&fldD[nr0], d0);
    atomicAdd(&fld[nr1], pv * d1);
    atomicAdd(&fldD[nr1], d1);
    atomicAdd(&fld[nr2], pv * d2);
    atomicAdd(&fldD[nr2], d2);
    atomicAdd(&fld[nr3], pv * d3);
    atomicAdd(&fldD[nr3], d3);
}

// === p2gNormalize_kernel ===
__global__ void p2gNormalize_kernel(float* fld, const float* fldD, int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    if (fldD[i] > 0.0f) fld[i] /= fldD[i];
}

// === restoreSolidCells_kernel ===
__global__ void restoreSolidCells_kernel(
    float* u, float* v,
    const float* prevU, const float* prevV,
    const int* cellType,
    int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= fNumX || j >= fNumY) return;

    int n = fNumY;
    int idx = i * n + j;
    bool solid = (cellType[idx] == SOLID_CELL);

    if (solid || (i > 0 && cellType[(i - 1) * n + j] == SOLID_CELL))
        u[idx] = prevU[idx];
    if (solid || (j > 0 && cellType[i * n + j - 1] == SOLID_CELL))
        v[idx] = prevV[idx];
}

// === updateParticleDensity_kernel ===
__global__ void updateParticleDensity_kernel(
    float* particleDensity,
    const float* posX, const float* posY,
    int numParticles,
    float h, float fInvSpacing,
    int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int n = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * hh;

    float x = clampf_d(posX[i], hh, (fNumX - 1) * hh);
    float y = clampf_d(posY[i], hh, (fNumY - 1) * hh);

    int x0 = (int)floorf((x - h2) * h1);
    float tx = ((x - h2) - x0 * hh) * h1;
    int x1 = min(x0 + 1, fNumX - 2);

    int y0 = (int)floorf((y - h2) * h1);
    float ty = ((y - h2) - y0 * hh) * h1;
    int y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx;
    float sy = 1.0f - ty;

    if (x0 < fNumX && y0 < fNumY) atomicAdd(&particleDensity[x0 * n + y0], sx * sy);
    if (x1 < fNumX && y0 < fNumY) atomicAdd(&particleDensity[x1 * n + y0], tx * sy);
    if (x1 < fNumX && y1 < fNumY) atomicAdd(&particleDensity[x1 * n + y1], tx * ty);
    if (x0 < fNumX && y1 < fNumY) atomicAdd(&particleDensity[x0 * n + y1], sx * ty);
}

// === launchSavePrevVelocities ===
void launchSavePrevVelocities(DeviceData& d, int fNumCells)
{
    savePrevVelocities_kernel<<<(fNumCells + 255) / 256, 256>>>(
        d.u, d.v, d.du, d.dv, d.prevU, d.prevV, fNumCells);
}

// === launchClassifyCells ===
void launchClassifyCells(DeviceData& d, int fNumCells, int numParticles,
    float fInvSpacing, int fNumX, int fNumY)
{
    classifyCellsA_kernel<<<(fNumCells + 255) / 256, 256>>>(d.cellType, d.s, fNumCells);
    classifyCellsB_kernel<<<(numParticles + 255) / 256, 256>>>(
        d.cellType, d.posX, d.posY, numParticles, fInvSpacing, fNumX, fNumY);
}

// === launchP2G ===
void launchP2G(DeviceData& d, int numParticles, int fNumCells,
    float h, float fInvSpacing, int fNumX, int fNumY)
{
    int grid = (numParticles + 255) / 256;
    p2gScatter_kernel<<<grid, 256>>>(d.u, d.du, d.posX, d.posY, d.velX,
        numParticles, 0, h, fInvSpacing, fNumX, fNumY);
    p2gScatter_kernel<<<grid, 256>>>(d.v, d.dv, d.posX, d.posY, d.velY,
        numParticles, 1, h, fInvSpacing, fNumX, fNumY);
    int gridC = (fNumCells + 255) / 256;
    p2gNormalize_kernel<<<gridC, 256>>>(d.u, d.du, fNumCells);
    p2gNormalize_kernel<<<gridC, 256>>>(d.v, d.dv, fNumCells);
}

// === launchRestoreSolidCells ===
void launchRestoreSolidCells(DeviceData& d, int fNumX, int fNumY)
{
    dim3 block(16, 16);
    dim3 grid((fNumX + 15) / 16, (fNumY + 15) / 16);
    restoreSolidCells_kernel<<<grid, block>>>(
        d.u, d.v, d.prevU, d.prevV, d.cellType, fNumX, fNumY);
}

// === launchUpdateParticleDensity ===
void launchUpdateParticleDensity(DeviceData& d, int numParticles, int fNumCells,
    float h, float fInvSpacing, int fNumX, int fNumY)
{
    cudaMemset(d.particleDensity, 0, fNumCells * sizeof(float));
    updateParticleDensity_kernel<<<(numParticles + 255) / 256, 256>>>(
        d.particleDensity, d.posX, d.posY, numParticles, h, fInvSpacing, fNumX, fNumY);
}
