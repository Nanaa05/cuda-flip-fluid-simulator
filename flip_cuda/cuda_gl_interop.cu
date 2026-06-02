// flip_cuda/cuda_gl_interop.cu
// NODE I2 (Brian) -- T10_interop, zero-copy CUDA-OpenGL

#include "flip_fluid.cuh"
#include "device_data.cuh"
#include "gl_render_pipeline.h"
#include <cuda_gl_interop.h>
#include <GL/gl.h>

// Mendaftarkan 5 buffer VBO OpenGL ke dalam konteks kerja CUDA
void interopInit(DeviceData& d, const RenderPipeline& rp) {
    // Gunakan flag WriteDiscard karena CUDA hanya bertugas menimpa/menulis data simulasi baru
    cudaGraphicsGLRegisterBuffer(&d.resource_posX,   rp.vboPosX,   cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&d.resource_posY,   rp.vboPosY,   cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&d.resource_colorR, rp.vboColorR, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&d.resource_colorG, rp.vboColorG, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&d.resource_colorB, rp.vboColorB, cudaGraphicsRegisterFlagsWriteDiscard);
}

// Memetakan resource VBO grafis dan mengambil raw pointer target di VRAM
void interopMapResources(DeviceData& d) {
    cudaGraphicsResource_t resources[] = {
        d.resource_posX,
        d.resource_posY,
        d.resource_colorR,
        d.resource_colorG,
        d.resource_colorB
    };
    
    // Map ke-5 resource secara kolektif dalam satu batch call demi efisiensi
    cudaGraphicsMapResources(5, resources, 0);

    // Ambil alamat raw device pointer dari masing-masing VBO grafis
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&d.posX,   &size, d.resource_posX);
    cudaGraphicsResourceGetMappedPointer((void**)&d.posY,   &size, d.resource_posY);
    cudaGraphicsResourceGetMappedPointer((void**)&d.colorR, &size, d.resource_colorR);
    cudaGraphicsResourceGetMappedPointer((void**)&d.colorG, &size, d.resource_colorG);
    cudaGraphicsResourceGetMappedPointer((void**)&d.colorB, &size, d.resource_colorB);
}

// Melepaskan kepemilikan buffer memori kembali ke OpenGL untuk digambar ke layar
void interopUnmapResources(DeviceData& d) {
    cudaGraphicsResource_t resources[] = {
        d.resource_posX,
        d.resource_posY,
        d.resource_colorR,
        d.resource_colorG,
        d.resource_colorB
    };
    
    cudaGraphicsUnmapResources(5, resources, 0);

    // Set pointer ke nullptr untuk mencegah modifikasi ilegal saat status unmapped
    d.posX   = nullptr;
    d.posY   = nullptr;
    d.colorR = nullptr;
    d.colorG = nullptr;
    d.colorB = nullptr;
}

// Membatalkan registrasi interop sebelum sistem dibersihkan atau shutdown
void interopDestroy(DeviceData& d) {
    if (d.resource_posX)   cudaGraphicsUnregisterResource(d.resource_posX);
    if (d.resource_posY)   cudaGraphicsUnregisterResource(d.resource_posY);
    if (d.resource_colorR) cudaGraphicsUnregisterResource(d.resource_colorR);
    if (d.resource_colorG) cudaGraphicsUnregisterResource(d.resource_colorG);
    if (d.resource_colorB) cudaGraphicsUnregisterResource(d.resource_colorB);

    d.resource_posX   = nullptr;
    d.resource_posY   = nullptr;
    d.resource_colorR = nullptr;
    d.resource_colorG = nullptr;
    d.resource_colorB = nullptr;
}