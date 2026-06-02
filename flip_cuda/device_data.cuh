// flip_cuda/device_data.cuh
// All device pointers in one struct. One instance lives on the host.

#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct DeviceData {
    // Particles (maxParticles each)
    float* posX   = nullptr;
    float* posY   = nullptr;
    float* velX   = nullptr;
    float* velY   = nullptr;
    float* colorR = nullptr;
    float* colorG = nullptr;
    float* colorB = nullptr;

    // MAC grid (fNumCells each)
    float* u              = nullptr;
    float* v              = nullptr;
    float* du             = nullptr;
    float* dv             = nullptr;
    float* prevU          = nullptr;
    float* prevV          = nullptr;
    float* p              = nullptr;
    float* s              = nullptr;
    int*   cellType       = nullptr;
    float* cellColor      = nullptr;  // 3 * fNumCells
    float* particleDensity = nullptr;

    // Spatial hash (pNumCells)
    int* numCellParticles  = nullptr;
    int* firstCellParticle = nullptr;  // pNumCells + 1
    int* cellParticleIds   = nullptr;  // maxParticles

    // Reduction scratch
    float* d_restDensity = nullptr;   // single float
    float* d_partialSums = nullptr;

    // GL interop handles (Node I2)
    cudaGraphicsResource_t vbo_pos_resource = nullptr;
    cudaGraphicsResource_t vbo_col_resource = nullptr;

    void allocate(int fNumCells, int pNumCells, int maxParticles);
    void free();
    void reset(const float* hPosX, const float* hPosY,
               const float* hS, int numParticles, int fNumCells);
};
