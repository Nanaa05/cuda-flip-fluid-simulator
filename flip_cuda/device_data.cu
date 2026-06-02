#include "device_data.cuh"
#include <vector>

// === DeviceData::allocate ===
void DeviceData::allocate(int fNumCells_, int pNumCells, int maxParticles_) {
    maxParticles = maxParticles_;
    fNumCells = fNumCells_;

    cudaMalloc(&velX, maxParticles * sizeof(float));
    cudaMalloc(&velY, maxParticles * sizeof(float));

    cudaMalloc(&u, fNumCells * sizeof(float));
    cudaMalloc(&v, fNumCells * sizeof(float));
    cudaMalloc(&du, fNumCells * sizeof(float));
    cudaMalloc(&dv, fNumCells * sizeof(float));
    cudaMalloc(&prevU, fNumCells * sizeof(float));
    cudaMalloc(&prevV, fNumCells * sizeof(float));
    cudaMalloc(&p, fNumCells * sizeof(float));
    cudaMalloc(&s, fNumCells * sizeof(float));
    cudaMalloc(&cellType, fNumCells * sizeof(int));
    cudaMalloc(&cellColor, fNumCells * 3 * sizeof(float));
    cudaMalloc(&particleDensity, fNumCells * sizeof(float));

    cudaMalloc(&numCellParticles, pNumCells * sizeof(int));
    cudaMalloc(&firstCellParticle, (pNumCells + 1) * sizeof(int));
    cudaMalloc(&cellParticleIds, maxParticles * sizeof(int));

    cudaMalloc(&d_restDensity, sizeof(float));
    cudaMalloc(&d_partialSums, (pNumCells + 1) * sizeof(int));
}

// === DeviceData::free ===
void DeviceData::free() {
    if (velX) cudaFree(velX);
    if (velY) cudaFree(velY);
    if (u) cudaFree(u);
    if (v) cudaFree(v);
    if (du) cudaFree(du);
    if (dv) cudaFree(dv);
    if (prevU) cudaFree(prevU);
    if (prevV) cudaFree(prevV);
    if (p) cudaFree(p);
    if (s) cudaFree(s);
    if (cellType) cudaFree(cellType);
    if (cellColor) cudaFree(cellColor);
    if (particleDensity) cudaFree(particleDensity);
    if (numCellParticles) cudaFree(numCellParticles);
    if (firstCellParticle) cudaFree(firstCellParticle);
    if (cellParticleIds) cudaFree(cellParticleIds);
    if (d_restDensity) cudaFree(d_restDensity);
    if (d_partialSums) cudaFree(d_partialSums);

    velX = velY = nullptr;
    u = v = du = dv = prevU = prevV = p = s = nullptr;
    cellType = nullptr;
    cellColor = particleDensity = nullptr;
    numCellParticles = firstCellParticle = cellParticleIds = nullptr;
    d_restDensity = d_partialSums = nullptr;
}

// === DeviceData::reset: map interop VBOs, copy initial positions and colors, unmap ===
void DeviceData::reset(const float* hPosX, const float* hPosY,
                       const float* hS, int numParticles, int fNumCells_) {
    cudaGraphicsResource_t resources[2] = { vbo_pos_resource, vbo_col_resource };
    cudaGraphicsMapResources(2, resources, 0);

    size_t sz;
    float* devPos = nullptr;
    float* devCol = nullptr;
    cudaGraphicsResourceGetMappedPointer((void**)&devPos, &sz, vbo_pos_resource);
    cudaGraphicsResourceGetMappedPointer((void**)&devCol, &sz, vbo_col_resource);

    posX = devPos;
    posY = devPos + maxParticles;
    colorR = devCol;
    colorG = devCol + maxParticles;
    colorB = devCol + 2 * maxParticles;

    cudaMemcpy(posX, hPosX, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(posY, hPosY, numParticles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(colorR, 0, numParticles * sizeof(float));
    cudaMemset(colorG, 0, numParticles * sizeof(float));
    std::vector<float> blueInit(numParticles, 1.0f);
    cudaMemcpy(colorB, blueInit.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);

    cudaGraphicsUnmapResources(2, resources, 0);
    posX = posY = colorR = colorG = colorB = nullptr;

    cudaMemset(velX, 0, numParticles * sizeof(float));
    cudaMemset(velY, 0, numParticles * sizeof(float));
    cudaMemcpy(s, hS, fNumCells_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(u, 0, fNumCells_ * sizeof(float));
    cudaMemset(v, 0, fNumCells_ * sizeof(float));
    cudaMemset(p, 0, fNumCells_ * sizeof(float));
}

extern "C" void launchDeviceDataReset(DeviceData& d, const float* hPosX, const float* hPosY,
                                      const float* hS, int numParticles, int fNumCells) {
    d.reset(hPosX, hPosY, hS, numParticles, fNumCells);
}
