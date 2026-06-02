// flip_cuda/flip_fluid.cuh
// Shared constants and SimParams uploaded to __constant__ memory.

#pragma once
#include <cuda_runtime.h>

constexpr int FLUID_CELL = 0;
constexpr int AIR_CELL   = 1;
constexpr int SOLID_CELL = 2;

// Uploaded once via cudaMemcpyToSymbol at scene setup.
struct SimParams {
    int   fNumX, fNumY, fNumCells;
    float h, fInvSpacing;
    int   pNumX, pNumY, pNumCells;
    float pInvSpacing;
    float particleRadius;
    float density;
    int   maxParticles;
};

extern __constant__ SimParams d_params;
