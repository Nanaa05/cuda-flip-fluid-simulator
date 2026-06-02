// flip_cuda/device_data.cuh
// Single source of truth untuk semua device pointer di GPU VRAM.
// Satu instance global dari struct ini akan hidup di sisi Host (CPU).

#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct DeviceData {
    // Ukuran alokasi untuk kalkulasi offset pointer pada skema 2-Resource Interop (Node I2)
    int maxParticles = 0;
    int fNumCells    = 0;

    // Pointer Partikel SoA (Nathanael & tim tidak perlu merombak kernel mereka)
    // Pointer posX, posY, colorR, colorG, colorB akan mengarah langsung ke memori VBO yang di-map
    float* posX   = nullptr;
    float* posY   = nullptr;
    float* velX   = nullptr;
    float* velY   = nullptr;
    float* colorR = nullptr;
    float* colorG = nullptr;
    float* colorB = nullptr;

    // MAC grid (masing-masing berukuran fNumCells)
    float* u              = nullptr;
    float* v              = nullptr;
    float* du             = nullptr;
    float* dv             = nullptr;
    float* prevU          = nullptr;
    float* prevV          = nullptr;
    float* p              = nullptr;
    float* s              = nullptr;
    int* cellType       = nullptr;
    float* cellColor      = nullptr;  // Berukuran 3 * fNumCells (RGB)
    float* particleDensity = nullptr;

    // Spatial hash (berukuran pNumCells)
    int* numCellParticles  = nullptr;
    int* firstCellParticle = nullptr;  // pNumCells + 1
    int* cellParticleIds   = nullptr;  // maxParticles

    // Scratchpad Memori untuk Reduksi Paralel
    float* d_restDensity = nullptr;   // Nilai rest density tunggal di GPU
    float* d_partialSums = nullptr;

    // GL interop handles (Node I2 - Menggunakan persis 2 Resource VBO)
    cudaGraphicsResource_t vbo_pos_resource = nullptr; // Menampung gabungan posisi X dan Y
    cudaGraphicsResource_t vbo_col_resource = nullptr; // Menampung gabungan warna R, G, B

    // Fungsi Siklus Hidup Memori GPU
    void allocate(int fNumCells, int pNumCells, int maxParticles);
    void free();
    void reset(const float* hPosX, const float* hPosY,
               const float* hS, int numParticles, int fNumCells);
};