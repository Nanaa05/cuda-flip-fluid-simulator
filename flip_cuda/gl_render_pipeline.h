// flip_cuda/gl_render_pipeline.h
// NODE I1 (Fachriza) -- modern OpenGL VBO pipeline

#pragma once
#include <GL/gl.h>

struct RenderPipeline {
    GLuint particleVAO = 0;
    GLuint particleVBO = 0;  // float2 positions, maxParticles
    GLuint colorVBO    = 0;  // float3 colors, maxParticles
    GLuint gridVAO     = 0;
    GLuint gridVBO     = 0;  // float3 per cell color, fNumCells
    GLuint particleShader = 0;
    GLuint gridShader     = 0;
    GLuint obstacleShader = 0;
    int maxParticles = 0;
    int fNumCells    = 0;
};

// renderInit: allocate VAOs, VBOs (GL_DYNAMIC_DRAW), compile shaders
//   particleVBO and colorVBO must exist before interopInit() in Node I2
void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells);

// renderParticles: glDrawArrays(GL_POINTS, 0, numParticles)
//   without interop: glBufferSubData positions/colors from host
//   with interop: skip upload, CUDA already wrote into VBO
void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx);

// renderGrid: upload cellColor[] via glBufferSubData, draw quads per cell
void renderGrid(RenderPipeline& rp, const float* cellColor, int fNumX, int fNumY, float h);

// renderObstacle: triangle fan via uniform center + radius
void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius);

// renderDestroy: glDeleteBuffers, glDeleteVertexArrays, glDeleteProgram
//   call before interopDestroy() on shutdown
void renderDestroy(RenderPipeline& rp);
