#pragma once
#include <GL/gl.h>

struct RenderPipeline {
    GLuint particleVAO = 0;
    GLuint particleVBO = 0;
    GLuint colorVBO = 0;

    GLuint gridVAO = 0;
    GLuint gridQuadVBO = 0;
    GLuint gridColorVBO = 0;

    GLuint obstacleVAO = 0;
    GLuint obstacleVBO = 0;
    int obstacleVertCount = 0;

    GLuint particleShader = 0;
    GLuint gridShader = 0;
    GLuint obstacleShader = 0;

    int maxParticles = 0;
    int fNumCells = 0;
    float simWidth = 0.0f;
    float simHeight = 0.0f;
    float projMatrix[16];
};

// === renderInit: allocate VAOs/VBOs, compile shaders, build ortho projection ===
void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells,
                int fNumX, int fNumY, float h);

// === renderParticles: draw GL_POINTS from interop VBOs, no host upload ===
void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx);

// === renderGrid: upload cellColor RGB host array, instanced quad draw ===
void renderGrid(RenderPipeline& rp, const float* cellColor, int fNumX, int fNumY, float h);

// === renderObstacle: triangle fan scaled by radius at (ox, oy) ===
void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius);

// === renderDestroy: delete all GL objects ===
void renderDestroy(RenderPipeline& rp);
