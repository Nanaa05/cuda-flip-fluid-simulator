// flip_cuda/device_data.cu
// Implementasi alokasi cerdas untuk mencegah double-allocation memori dan reset aman dengan interop

#include "device_data.cuh"
#include <vector>
#include <cstdio>

void DeviceData::allocate(int fNumCells_, int pNumCells, int maxParticles_) {
    maxParticles = maxParticles_;
    fNumCells = fNumCells_;

    // Pointers posX, posY, colorR, colorG, colorB TIDAK dialokasikan lewat cudaMalloc
    // karena memori fisiknya dimiliki oleh OpenGL VBO dan akan dipetakan (mapped) secara dinamis.
    cudaMalloc(&velX, maxParticles * sizeof(float));
    cudaMalloc(&velY, maxParticles * sizeof(float));

    // Alokasi memori untuk variabel MAC Grid
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

    // Alokasi memori untuk sistem Spatial Hashing
    cudaMalloc(&numCellParticles, pNumCells * sizeof(int));
    cudaMalloc(&firstCellParticle, (pNumCells + 1) * sizeof(int));
    cudaMalloc(&cellParticleIds, maxParticles * sizeof(int));

    // Alokasi workspace untuk reduksi paralel
    cudaMalloc(&d_restDensity, sizeof(float));
    cudaMalloc(&d_partialSums, (pNumCells + 1) * sizeof(int)); // Digunakan kembali untuk CUB scan
}

void DeviceData::free() {
    // Bebaskan memori partikel
    if (velX) cudaFree(velX);
    if (velY) cudaFree(velY);

    // Bebaskan memori MAC Grid
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

    // Bebaskan memori Spatial Hashing
    if (numCellParticles) cudaFree(numCellParticles);
    if (firstCellParticle) cudaFree(firstCellParticle);
    if (cellParticleIds) cudaFree(cellParticleIds);

    // Bebaskan workspace reduksi
    if (d_restDensity) cudaFree(d_restDensity);
    if (d_partialSums) cudaFree(d_partialSums);

    // Setel ulang semua pointer ke status null untuk keamanan
    velX = velY = nullptr;
    u = v = du = dv = prevU = prevV = p = s = nullptr;
    cellType = nullptr; cellColor = particleDensity = nullptr;
    numCellParticles = firstCellParticle = cellParticleIds = nullptr;
    d_restDensity = d_partialSums = nullptr;
}

void DeviceData::reset(const float* hPosX, const float* hPosY,
                       const float* hS, int numParticles, int fNumCells) 
{
    // POLA SAFETY RESET INTEROP:
    // Selama pemanggilan setupScene() di main, resource OpenGL VBO harus dipetakan sementara
    // agar pointer d.posX dkk mendapatkan alamat virtual VRAM yang valid sebelum disalin.
    cudaGraphicsResource_t resources[] = { vbo_pos_resource, vbo_col_resource };
    cudaGraphicsMapResources(2, resources, 0);
    
    size_t size;
    float* dev_pos = nullptr;
    float* dev_col = nullptr;
    cudaGraphicsResourceGetMappedPointer((void**)&dev_pos, &size, vbo_pos_resource);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_col, &size, vbo_col_resource);
    
    // Terapkan pembagian offset biner linear
    posX = dev_pos;
    posY = dev_pos + maxParticles;
    colorR = dev_col;
    colorG = dev_col + maxParticles;
    colorB = dev_col + (2 * maxParticles);

    // Salin koordinat awal dari Host langsung ke dalam alamat VBO grafis yang telah ter-map
    cudaMemcpy(posX, hPosX, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(posY, hPosY, numParticles * sizeof(float), cudaMemcpyHostToDevice);

    // Setel warna partikel awal (biru solid: R=0, G=0, B=1.0)
    cudaMemset(colorR, 0, numParticles * sizeof(float));
    cudaMemset(colorG, 0, numParticles * sizeof(float));
    std::vector<float> tempBlue(numParticles, 1.0f);
    cudaMemcpy(colorB, tempBlue.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);

    // Lepas pemetaan agar OpenGL kembali memegang kendali buffer sebelum loop utama dimulai
    cudaGraphicsUnmapResources(2, resources, 0);

    // Kosongkan pointer di host demi keamanan pointer
    posX = posY = colorR = colorG = colorB = nullptr;

    // Bersihkan sisa variabel grid dan kecepatan partikel sekunder
    cudaMemset(velX, 0, numParticles * sizeof(float));
    cudaMemset(velY, 0, numParticles * sizeof(float));
    cudaMemcpy(s, hS, fNumCells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(u, 0, fNumCells * sizeof(float));
    cudaMemset(v, 0, fNumCells * sizeof(float));
    cudaMemset(p, 0, fNumCells * sizeof(float));
}