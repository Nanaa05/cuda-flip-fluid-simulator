// flip_cuda/cuda_gl_interop.cu
// NODE I2 (Brian) -- T10_interop, zero-copy CUDA-OpenGL

#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cuda_gl_interop.h>
#include <GL/gl.h>

// interopInit(DeviceData& d, GLuint particleVBO, GLuint colorVBO)
// - cudaGraphicsGLRegisterBuffer(&d.vbo_pos_resource, particleVBO,
//                                cudaGraphicsRegisterFlagsWriteDiscard)
// - cudaGraphicsGLRegisterBuffer(&d.vbo_col_resource, colorVBO,
//                                cudaGraphicsRegisterFlagsWriteDiscard)
// - called once after renderInit() creates the VBOs

// interopMapResources(DeviceData& d)
// - cudaGraphicsMapResources for both resources
// - cudaGraphicsResourceGetMappedPointer -> d.posX, d.posY, d.colorR/G/B
// - after this, CUDA kernels write directly into GL VBO memory
// - measure this call as part of T10_interop

// interopUnmapResources(DeviceData& d)
// - cudaGraphicsUnmapResources for both resources
// - after this, GL owns the buffer again; d.posX etc. are invalid
// - call before renderParticles()

// interopDestroy(DeviceData& d)
// - cudaGraphicsUnregisterResource for both resources
// - call before renderDestroy() on scene reset or shutdown
