// flip_cuda/flip_fluid.cuh
// Shared constants and SimParams uploaded to __constant__ memory.

#pragma once
#include <cuda_runtime.h>

constexpr int FLUID_CELL = 0;
constexpr int AIR_CELL   = 1;
constexpr int SOLID_CELL = 2;

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

struct GpuTelemetry {
    double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
    double t6 = 0.0, t7 = 0.0, t8 = 0.0, t9 = 0.0, t10 = 0.0, t_total = 0.0;
    int frames = 0;
    int pressureIters = 0;

    void reset() {
        t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = t9 = t10 = t_total = 0.0;
        frames = 0;
    }
};

extern GpuTelemetry g_gpu_telemetry;
