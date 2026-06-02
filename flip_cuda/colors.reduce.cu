// flip_cuda/colors_reduce.cu
// T8 (Brian) -- updateParticleColors + updateCellColors + computeRestDensity

#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper device function untuk memetakan nilai densitas ke warna pelangi (Scientific Color Map)
__device__ void setSciColor_d(float* cellColor, int cellNr, float val, float minVal, float maxVal) {
    float val_clamped = fmaxf(minVal, fminf(maxVal, val));
    float dv = maxVal - minVal;
    float r = 1.0f, g = 1.0f, b = 1.0f;

    if (dv > 0.0f) {
        val_clamped = (val_clamped - minVal) / dv;
    } else {
        val_clamped = 0.5f;
    }

    // Skema pewarnaan Jet (Biru -> Hijau -> Merah)
    if (val_clamped < 0.25f) {
        r = 0.0f;
        g = 4.0f * val_clamped;
    } else if (val_clamped < 0.5f) {
        r = 0.0f;
        b = 1.0f - 4.0f * (val_clamped - 0.25f);
    } else if (val_clamped < 0.75f) {
        r = 4.0f * (val_clamped - 0.5f);
        b = 0.0f;
    } else {
        g = 1.0f - 4.0f * (val_clamped - 0.75f);
        b = 0.0f;
    }

    cellColor[3 * cellNr + 0] = r;
    cellColor[3 * cellNr + 1] = g;
    cellColor[3 * cellNr + 2] = b;
}

// Kernel reduksi paralel sederhana untuk menjumlahkan densitas sel fluida
__global__ void computeRestDensity_kernel(
    const float* particleDensity, const int* cellType, 
    float* d_partialSums, int* d_fluidCount, int fNumCells) 
{
    // Shared memory untuk reduksi di dalam satu block thread
    extern __shared__ float s_data[];
    __shared__ int s_count[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    int count = 0;

    // Membaca data secara coalesced ke register thread
    if (i < fNumCells) {
        if (cellType[i] == FLUID_CELL) {
            sum = particleDensity[i];
            count = 1;
        }
    }

    s_data[tid] = sum;
    s_count[tid] = count;
    __syncthreads();

    // Reduksi paralel di dalam shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }

    // Tulis hasil parsial block ke memori global
    if (tid == 0) {
        d_partialSums[blockIdx.x] = s_data[0];
        atomicAdd(d_fluidCount, s_count[0]);
    }
}

// Kernel untuk memperbarui gradasi warna partikel berdasarkan kepadatan lokal di VRAM
__global__ void updateParticleColors_kernel(
    float* colorR, float* colorG, float* colorB,
    const float* posX, const float* posY, const float* particleDensity,
    float restDensity, int numParticles, float pInvSpacing, int pNumX, int pNumY, int fNumY) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    // Hitung posisi koordinat grid spasial partikel
    int xi = (int)(posX[i] * pInvSpacing);
    int yi = (int)(posY[i] * pInvSpacing);
    
    // Cari cellNr grid simulasi yang menaungi partikel tersebut
    int cellNr = xi * fNumY + yi;
    float densityRatio = 1.0f;
    
    if (restDensity > 0.0f) {
        densityRatio = particleDensity[cellNr] / restDensity;
    }

    // Skema transisi warna partikel berbasis gaya hibrida air (biru pekat ke putih berbusa)
    if (densityRatio < 0.7f) {
        // Efek busa / percikan air (Warna putih terang kebiruan)
        colorR[i] = 0.8f;
        colorG[i] = 0.8f;
        colorB[i] = 1.0f;
    } else {
        // Efek aliran air normal (Pergeseran gradasi drift seiring waktu)
        float r = colorR[i] - 0.01f;
        float g = colorG[i] - 0.01f;
        float b = colorB[i] + 0.01f;

        colorR[i] = fmaxf(0.0f, fminf(1.0f, r));
        colorG[i] = fmaxf(0.0f, fminf(1.0f, g));
        colorB[i] = fmaxf(0.0f, fminf(1.0f, b));
    }
}

// Kernel untuk merender visualisasi sel grid (Udara, Fluida, Padat/Obstacle)
__global__ void updateCellColors_kernel(
    float* cellColor, const int* cellType, 
    const float* particleDensity, float restDensity, int fNumCells) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;

    int type = cellType[i];

    if (type == SOLID_CELL) {
        // SOLID: Sel rintangan diwarnai abu-abu gelap
        cellColor[3 * i + 0] = 0.5f;
        cellColor[3 * i + 1] = 0.5f;
        cellColor[3 * i + 2] = 0.5f;
    } else if (type == FLUID_CELL) {
        // FLUID: Petakan densitas lokal ke gradasi warna pelangi (Jet)
        float val = particleDensity[i];
        setSciColor_d(cellColor, i, val, 0.0f, 2.0f * restDensity);
    } else {
        // AIR: Sel udara kosong diberi warna hitam pekat agar kontras
        cellColor[3 * i + 0] = 0.0f;
        cellColor[3 * i + 1] = 0.0f;
        cellColor[3 * i + 2] = 0.0f;
    }
}

// Wrapper Host: Menghitung rata-rata densitas sel fluida (restDensity) secara paralel
float launchComputeRestDensity(DeviceData& d, int fNumCells) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (fNumCells + threadsPerBlock - 1) / threadsPerBlock;

    // Alokasikan variabel bantu di GPU untuk menghitung jumlah sel aktif
    int* d_fluidCount = nullptr;
    cudaMalloc(&d_fluidCount, sizeof(int));
    cudaMemset(d_fluidCount, 0, sizeof(int));

    // Eksekusi kernel reduksi paralel tahap pertama
    computeRestDensity_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d.particleDensity, d.cellType, d.d_partialSums, d_fluidCount, fNumCells
    );

    // Ambil hasil penjumlahan parsial dan jumlah sel fluida aktif kembali ke Host
    std::vector<float> h_partialSums(blocksPerGrid);
    cudaMemcpy(h_partialSums.data(), d.d_partialSums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    int h_fluidCount = 0;
    cudaMemcpy(&h_fluidCount, d_fluidCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_fluidCount);

    // Selesaikan akumulasi akhir reduksi di sisi Host
    float totalDensitySum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        totalDensitySum += h_partialSums[i];
    }

    if (h_fluidCount > 0) {
        return totalDensitySum / (float)h_fluidCount;
    }
    return 0.0f; // Default fallback jika tidak ada sel fluida terdeteksi
}

// Wrapper Host: Menjalankan pembaruan gradasi warna partikel fluida
void launchUpdateParticleColors(DeviceData& d, int numParticles) {
    if (numParticles <= 0) return;

    // Ambil parameter dari Constant Memory GPU
    float pInvSpacing;
    int pNumX, pNumY, fNumY;
    float restDensity;

    cudaMemcpyFromSymbol(&pInvSpacing, d_params, sizeof(float), offsetof(SimParams, pInvSpacing), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&pNumX, d_params, sizeof(int), offsetof(SimParams, pNumX), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&pNumY, d_params, sizeof(int), offsetof(SimParams, pNumY), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&fNumY, d_params, sizeof(int), offsetof(SimParams, fNumY), cudaMemcpyDeviceToHost);
    
    // Tarik nilai rest density aktif dari VRAM
    cudaMemcpy(&restDensity, d.d_restDensity, sizeof(float), cudaMemcpyDeviceToHost);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    updateParticleColors_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d.colorR, d.colorG, d.colorB,
        d.posX, d.posY, d.particleDensity,
        restDensity, numParticles, pInvSpacing, pNumX, pNumY, fNumY
    );
}

// Wrapper Host: Menjalankan pembaruan warna sel grid makro
void launchUpdateCellColors(DeviceData& d, int fNumCells) {
    if (fNumCells <= 0) return;

    float restDensity;
    cudaMemcpy(&restDensity, d.d_restDensity, sizeof(float), cudaMemcpyDeviceToHost);

    int threadsPerBlock = 256;
    int blocksPerGrid = (fNumCells + threadsPerBlock - 1) / threadsPerBlock;

    updateCellColors_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d.cellColor, d.cellType, d.particleDensity, restDensity, fNumCells
    );
}