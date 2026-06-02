#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct DeviceData {
    int maxParticles = 0;
    int fNumCells = 0;

    float* posX = nullptr;
    float* posY = nullptr;
    float* velX = nullptr;
    float* velY = nullptr;
    float* colorR = nullptr;
    float* colorG = nullptr;
    float* colorB = nullptr;

    float* u = nullptr;
    float* v = nullptr;
    float* du = nullptr;
    float* dv = nullptr;
    float* prevU = nullptr;
    float* prevV = nullptr;
    float* p = nullptr;
    float* s = nullptr;
    int* cellType = nullptr;
    float* cellColor = nullptr;
    float* particleDensity = nullptr;

    int* numCellParticles = nullptr;
    int* firstCellParticle = nullptr;
    int* cellParticleIds = nullptr;

    float* d_restDensity = nullptr;
    float* d_partialSums = nullptr;

    cudaGraphicsResource_t vbo_pos_resource = nullptr;
    cudaGraphicsResource_t vbo_col_resource = nullptr;

    void allocate(int fNumCells, int pNumCells, int maxParticles);
    void free();
    void reset(const float* hPosX, const float* hPosY,
               const float* hS, int numParticles, int fNumCells);
};

extern "C" void launchDeviceDataReset(DeviceData& d, const float* hPosX, const float* hPosY,
                                      const float* hS, int numParticles, int fNumCells);
