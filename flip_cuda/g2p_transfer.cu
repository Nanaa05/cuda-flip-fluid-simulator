#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cuda_runtime.h>

static __device__ __forceinline__ float clampf_d(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

__global__ void g2pGather_kernel(float* particleVel,
                                  const float* posX, const float* posY,
                                  const float* fld, const float* pfld,
                                  const int* cellType,
                                  int numParticles, int component, float flipRatio,
                                  float h, float fInvSpacing, int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int n = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * hh;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2 : 0.0f;
    int offset = (component == 0) ? n : 1;

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

    float valid0 = (cellType[nr0] != AIR_CELL || cellType[nr0 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid1 = (cellType[nr1] != AIR_CELL || cellType[nr1 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid2 = (cellType[nr2] != AIR_CELL || cellType[nr2 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid3 = (cellType[nr3] != AIR_CELL || cellType[nr3 - offset] != AIR_CELL) ? 1.0f : 0.0f;

    float d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;

    if (d > 0.0f) {
        float f0 = fld[nr0], f1 = fld[nr1], f2 = fld[nr2], f3 = fld[nr3];
        float pf0 = pfld[nr0], pf1 = pfld[nr1], pf2 = pfld[nr2], pf3 = pfld[nr3];

        float picV = (valid0*d0*f0 + valid1*d1*f1 + valid2*d2*f2 + valid3*d3*f3) / d;
        float corr = (valid0*d0*(f0-pf0) + valid1*d1*(f1-pf1)
                    + valid2*d2*(f2-pf2) + valid3*d3*(f3-pf3)) / d;
        float flipV = particleVel[i] + corr;
        particleVel[i] = (1.0f - flipRatio) * picV + flipRatio * flipV;
    }
}

void launchG2P(DeviceData& d, int numParticles, float flipRatio,
               float h, float fInvSpacing, int fNumX, int fNumY)
{
    int grid = (numParticles + 255) / 256;
    g2pGather_kernel<<<grid, 256>>>(d.velX, d.posX, d.posY, d.u, d.prevU,
                                     d.cellType, numParticles, 0, flipRatio,
                                     h, fInvSpacing, fNumX, fNumY);
    g2pGather_kernel<<<grid, 256>>>(d.velY, d.posX, d.posY, d.v, d.prevV,
                                     d.cellType, numParticles, 1, flipRatio,
                                     h, fInvSpacing, fNumX, fNumY);
}
