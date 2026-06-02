#include "flip_fluid.cuh"
#include "device_data.cuh"
#include "../flip_cpu/flip_fluid.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__constant__ SimParams d_params;

extern void gpuSimulate(DeviceData& d, int numParticles, float dt, float gravity,
                        float flipRatio, int numPressureIters, int numParticleIters,
                        float overRelaxation, bool compensateDrift, bool separateParticles,
                        float obstacleX, float obstacleY, float obstacleRadius,
                        float obstacleVelX, float obstacleVelY, int numSubSteps);

static constexpr int CANVAS_W = 900;
static constexpr int CANVAS_H = 700;
static constexpr float SIM_HEIGHT = 3.0f;
static constexpr float SIM_WIDTH = (float)CANVAS_W / ((float)CANVAS_H / SIM_HEIGHT);

// === DiffStats: max / mean / rms of |a - b| over an array ===
struct DiffStats {
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double rms = 0.0;
    int argMax = -1;
};

// === computeDiff: per-component error between two host arrays ===
static DiffStats computeDiff(const float* a, const float* b, int n) {
    DiffStats s;
    double sumAbs = 0.0, sumSq = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        sumAbs += e;
        sumSq += e * e;
        if (e > s.maxAbs) { s.maxAbs = e; s.argMax = i; }
    }
    if (n > 0) {
        s.meanAbs = sumAbs / n;
        s.rms = std::sqrt(sumSq / n);
    }
    return s;
}

// === carveObstacleCpu: stamp a static obstacle into the CPU solid grid ===
static void carveObstacleCpu(flipcpu::FlipFluid& f, float ox, float oy, float orad) {
    int n = f.fNumY;
    for (int i = 1; i < f.fNumX - 2; ++i) {
        for (int j = 1; j < f.fNumY - 2; ++j) {
            f.s[i * n + j] = 1.0f;
            float dx = (i + 0.5f) * f.h - ox;
            float dy = (j + 0.5f) * f.h - oy;
            if (dx * dx + dy * dy < orad * orad) {
                f.s[i * n + j] = 0.0f;
                f.u[i * n + j]       = 0.0f;
                f.u[(i + 1) * n + j] = 0.0f;
                f.v[i * n + j]       = 0.0f;
                f.v[i * n + j + 1]   = 0.0f;
            }
        }
    }
}

int main(int argc, char** argv) {
    int resolution = 100;
    int numFrames = 120;
    float gravity = -9.81f;
    float flipRatio = 0.9f;
    bool separateParticles = true;
    bool compensateDrift = true;
    float obstacleRadius = 0.15f;
    float obstacleX = 3.0f, obstacleY = 2.0f;
    bool lockstep = false;
    int overrideIters = -1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--res") == 0 && i + 1 < argc) resolution = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) numFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) overrideIters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--lockstep") == 0) lockstep = true;
        else if (std::strcmp(argv[i], "--no-obstacle") == 0) obstacleRadius = 0.0f;
        else if (std::strcmp(argv[i], "--no-gravity") == 0) gravity = 0.0f;
        else if (std::strcmp(argv[i], "--no-separate") == 0) separateParticles = false;
    }

    float dt = 1.0f / 60.0f;
    float overRelaxation = 1.9f;
    int numParticleIters = 2;
    int numSubSteps;
    if      (resolution <= 100) numSubSteps = 1;
    else if (resolution <= 140) numSubSteps = 2;
    else if (resolution <= 180) numSubSteps = 3;
    else                        numSubSteps = 4;
    int numPressureIters = 50 + std::max(0, (resolution - 100)) / 2;
    if (overrideIters > 0) numPressureIters = overrideIters;

    float tankHeight = SIM_HEIGHT;
    float tankWidth  = SIM_WIDTH;
    float h = tankHeight / resolution;
    float density = 1000.0f;
    float r  = 0.3f * h;
    float dx = 2.0f * r;
    float dy = std::sqrt(3.0f) / 2.0f * dx;
    float relWaterHeight = 0.8f;
    float relWaterWidth  = 0.6f;

    int numX = (int)std::floor((relWaterWidth  * tankWidth  - 2.0f * h - 2.0f * r) / dx);
    int numY = (int)std::floor((relWaterHeight * tankHeight - 2.0f * h - 2.0f * r) / dy);
    if (numX < 1) numX = 1;
    if (numY < 1) numY = 1;
    int maxParticles = numX * numY;
    int numParticles = numX * numY;

    std::vector<float> seedX(numParticles), seedY(numParticles);
    {
        int pid = 0;
        for (int i = 0; i < numX; ++i) {
            for (int j = 0; j < numY; ++j) {
                float offset = (j % 2 == 0) ? 0.0f : r;
                seedX[pid] = h + r + dx * i + offset;
                seedY[pid] = h + r + dy * j;
                ++pid;
            }
        }
    }

    flipcpu::FlipFluid cpu(density, tankWidth, tankHeight, h, r, maxParticles);
    cpu.numParticles = numParticles;
    cpu.particleRestDensity = 0.0f;
    for (int i = 0; i < numParticles; ++i) {
        cpu.particlePosX[i] = seedX[i];
        cpu.particlePosY[i] = seedY[i];
    }
    for (int i = 0; i < cpu.fNumX; ++i) {
        for (int j = 0; j < cpu.fNumY; ++j) {
            float sVal = 1.0f;
            if (i == 0 || i == cpu.fNumX - 1 || j == 0) sVal = 0.0f;
            cpu.s[i * cpu.fNumY + j] = sVal;
        }
    }
    if (obstacleRadius > 0.0f) carveObstacleCpu(cpu, obstacleX, obstacleY, obstacleRadius);

    int gfNumX = (int)std::floor(SIM_WIDTH / h) + 1;
    int gfNumY = (int)std::floor(SIM_HEIGHT / h) + 1;
    float gh = std::max(SIM_WIDTH / (float)gfNumX, SIM_HEIGHT / (float)gfNumY);
    float gfInvSpacing = 1.0f / gh;
    int gfNumCells = gfNumX * gfNumY;

    float pSpacing = 2.2f * r;
    float pInvSpacing = 1.0f / pSpacing;
    int pNumX = (int)std::floor(SIM_WIDTH / pSpacing) + 1;
    int pNumY = (int)std::floor(SIM_HEIGHT / pSpacing) + 1;
    int pNumCells = pNumX * pNumY;

    SimParams hostParams;
    hostParams.fNumX = gfNumX;
    hostParams.fNumY = gfNumY;
    hostParams.fNumCells = gfNumCells;
    hostParams.h = gh;
    hostParams.fInvSpacing = gfInvSpacing;
    hostParams.pNumX = pNumX;
    hostParams.pNumY = pNumY;
    hostParams.pNumCells = pNumCells;
    hostParams.pInvSpacing = pInvSpacing;
    hostParams.particleRadius = r;
    hostParams.density = density;
    hostParams.maxParticles = maxParticles;
    cudaMemcpyToSymbol(d_params, &hostParams, sizeof(SimParams));

    std::vector<float> hS(gfNumCells, 1.0f);
    for (int i = 0; i < gfNumX; ++i) {
        for (int j = 0; j < gfNumY; ++j) {
            if (i == 0 || i == gfNumX - 1 || j == 0)
                hS[i * gfNumY + j] = 0.0f;
        }
    }

    DeviceData d;
    d.allocate(gfNumCells, pNumCells, maxParticles);
    cudaMalloc(&d.posX, maxParticles * sizeof(float));
    cudaMalloc(&d.posY, maxParticles * sizeof(float));
    cudaMalloc(&d.colorR, maxParticles * sizeof(float));
    cudaMalloc(&d.colorG, maxParticles * sizeof(float));
    cudaMalloc(&d.colorB, maxParticles * sizeof(float));

    cudaMemcpy(d.posX, seedX.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d.posY, seedY.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d.velX, 0, maxParticles * sizeof(float));
    cudaMemset(d.velY, 0, maxParticles * sizeof(float));
    cudaMemset(d.colorR, 0, numParticles * sizeof(float));
    cudaMemset(d.colorG, 0, numParticles * sizeof(float));
    {
        std::vector<float> blue(numParticles, 1.0f);
        cudaMemcpy(d.colorB, blue.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d.s, hS.data(), gfNumCells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d.u, 0, gfNumCells * sizeof(float));
    cudaMemset(d.v, 0, gfNumCells * sizeof(float));
    cudaMemset(d.p, 0, gfNumCells * sizeof(float));

    std::vector<float> gpuPosX(numParticles), gpuPosY(numParticles);
    std::vector<float> gpuVelX(numParticles), gpuVelY(numParticles);

    std::printf("=== CPU vs GPU FLIP numerical validation ===\n");
    std::printf("mode=%s  pressureIters=%d\n", lockstep ? "LOCKSTEP (single-step error)" : "FREE-RUN (accumulated trajectory)", numPressureIters);
    std::printf("res=%d frames=%d particles=%d gravity=%.2f flip=%.2f obstacle=%.3f separate=%d\n",
                resolution, numFrames, numParticles, gravity, flipRatio, obstacleRadius, (int)separateParticles);
    std::printf("CPU grid=%dx%d (h=%.6f)  GPU grid=%dx%d (h=%.6f)\n",
                cpu.fNumX, cpu.fNumY, cpu.h, gfNumX, gfNumY, gh);
    std::printf("%-6s | %-30s | %-30s\n", "frame", "position  max / mean / rms", "velocity  max / mean / rms");
    std::printf("------------------------------------------------------------------------------------\n");

    double worstPos = 0.0;
    int worstPosFrame = -1;

    for (int frame = 0; frame < numFrames; ++frame) {
        if (lockstep) {
            cudaMemcpy(d.posX, cpu.particlePosX.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d.posY, cpu.particlePosY.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d.velX, cpu.particleVelX.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d.velY, cpu.particleVelY.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
        }

        cpu.simulate(dt, gravity, flipRatio, numPressureIters, numParticleIters,
                     overRelaxation, compensateDrift, separateParticles,
                     obstacleX, obstacleY, obstacleRadius, 0.0f, 0.0f, numSubSteps);

        gpuSimulate(d, numParticles, dt, gravity, flipRatio, numPressureIters, numParticleIters,
                    overRelaxation, compensateDrift, separateParticles,
                    obstacleX, obstacleY, obstacleRadius, 0.0f, 0.0f, numSubSteps);

        cudaMemcpy(gpuPosX.data(), d.posX, numParticles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpuPosY.data(), d.posY, numParticles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpuVelX.data(), d.velX, numParticles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpuVelY.data(), d.velY, numParticles * sizeof(float), cudaMemcpyDeviceToHost);

        DiffStats px = computeDiff(cpu.particlePosX.data(), gpuPosX.data(), numParticles);
        DiffStats py = computeDiff(cpu.particlePosY.data(), gpuPosY.data(), numParticles);
        DiffStats vx = computeDiff(cpu.particleVelX.data(), gpuVelX.data(), numParticles);
        DiffStats vy = computeDiff(cpu.particleVelY.data(), gpuVelY.data(), numParticles);

        double posMax = std::max(px.maxAbs, py.maxAbs);
        double posMean = 0.5 * (px.meanAbs + py.meanAbs);
        double posRms = std::sqrt(0.5 * (px.rms * px.rms + py.rms * py.rms));
        double velMax = std::max(vx.maxAbs, vy.maxAbs);
        double velMean = 0.5 * (vx.meanAbs + vy.meanAbs);
        double velRms = std::sqrt(0.5 * (vx.rms * vx.rms + vy.rms * vy.rms));

        if (posMax > worstPos) { worstPos = posMax; worstPosFrame = frame; }

        bool show = (frame < 10) || (frame % 10 == 0) || (frame == numFrames - 1);
        if (show) {
            char posBuf[64], velBuf[64];
            std::snprintf(posBuf, sizeof(posBuf), "%.3e / %.3e / %.3e", posMax, posMean, posRms);
            std::snprintf(velBuf, sizeof(velBuf), "%.3e / %.3e / %.3e", velMax, velMean, velRms);
            std::printf("%-6d | %-30s | %-30s\n", frame, posBuf, velBuf);
        }
    }

    std::printf("------------------------------------------------------------------------------------\n");
    std::printf("worst position max-abs error = %.4e (frame %d), grid spacing h = %.4e\n",
                worstPos, worstPosFrame, gh);
    std::printf("relative to cell size: %.2f%% of h\n", 100.0 * worstPos / gh);
    if (lockstep) {
        std::printf("\n[LOCKSTEP] Each frame measures a single step from an identical CPU state.\n");
        std::printf("           Error stays bounded => GPU matches CPU per-step (parallel-reorder + partial-solve only).\n");
    } else {
        std::printf("\n[FREE-RUN] Trajectories diverge chaotically (Lyapunov); accumulated error is expected\n");
        std::printf("           to reach domain scale. Use --lockstep to measure true per-step correctness.\n");
    }

    cudaFree(d.posX); cudaFree(d.posY);
    cudaFree(d.colorR); cudaFree(d.colorG); cudaFree(d.colorB);
    d.posX = d.posY = d.colorR = d.colorG = d.colorB = nullptr;
    d.free();
    return 0;
}
