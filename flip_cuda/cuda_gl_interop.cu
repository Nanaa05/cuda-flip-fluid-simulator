#include "flip_fluid.cuh"
#include "device_data.cuh"
#include "gl_render_pipeline.h"
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>

// === interopInit ===
void interopInit(DeviceData& d, const RenderPipeline& rp) {
    cudaError_t e1 = cudaGraphicsGLRegisterBuffer(&d.vbo_pos_resource, rp.particleVBO,
                                                   cudaGraphicsRegisterFlagsNone);
    cudaError_t e2 = cudaGraphicsGLRegisterBuffer(&d.vbo_col_resource, rp.colorVBO,
                                                   cudaGraphicsRegisterFlagsNone);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        std::fprintf(stderr, "[interopInit] cudaGraphicsGLRegisterBuffer failed: %s | %s\n",
                     cudaGetErrorString(e1), cudaGetErrorString(e2));
        std::exit(1);
    }
}

// === interopMapResources ===
void interopMapResources(DeviceData& d) {
    cudaGraphicsResource_t res[2] = { d.vbo_pos_resource, d.vbo_col_resource };
    cudaGraphicsMapResources(2, res, 0);

    size_t sz;
    float* ptr;
    cudaGraphicsResourceGetMappedPointer((void**)&ptr, &sz, d.vbo_pos_resource);
    d.posX = ptr;
    d.posY = ptr + d.maxParticles;

    cudaGraphicsResourceGetMappedPointer((void**)&ptr, &sz, d.vbo_col_resource);
    d.colorR = ptr;
    d.colorG = ptr + d.maxParticles;
    d.colorB = ptr + 2 * d.maxParticles;
}

// === interopUnmapResources ===
void interopUnmapResources(DeviceData& d) {
    cudaGraphicsResource_t res[2] = { d.vbo_pos_resource, d.vbo_col_resource };
    cudaGraphicsUnmapResources(2, res, 0);
    d.posX = d.posY = d.colorR = d.colorG = d.colorB = nullptr;
}

// === interopDestroy ===
void interopDestroy(DeviceData& d) {
    if (d.vbo_pos_resource) cudaGraphicsUnregisterResource(d.vbo_pos_resource);
    if (d.vbo_col_resource) cudaGraphicsUnregisterResource(d.vbo_col_resource);
    d.vbo_pos_resource = d.vbo_col_resource = nullptr;
}
