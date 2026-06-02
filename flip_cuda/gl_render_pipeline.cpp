// flip_cuda/gl_render_pipeline.cpp
// NODE I1 (Fachriza) -- replaces legacy glBegin/glEnd with VBO pipeline

#include "gl_render_pipeline.h"
#include <GL/gl.h>
#include <cstdio>

// renderInit
// - glGenVertexArrays + glGenBuffers for particle and grid VAO/VBO
// - glBufferData with GL_DYNAMIC_DRAW for particleVBO, colorVBO, gridVBO
// - compile vertex + fragment shaders for particles, grid cells, obstacle

void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells) {
    rp.maxParticles = maxParticles;
    rp.fNumCells = fNumCells;
    rp.particleVBO = 1; 
    rp.colorVBO = 2;
    rp.gridVBO = 3;
    printf("[DUMMY] renderInit executed for %d particles and %d cells.\n", maxParticles, fNumCells);
}

// renderParticles
// - glBindVertexArray(particleVAO)
// - glUseProgram(particleShader), set pointSize uniform
// - glDrawArrays(GL_POINTS, 0, numParticles)

void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx) {

}

// renderGrid
// - glBufferSubData cellColor into gridVBO
// - draw: either instanced quads or pre-expanded 4-vertex-per-cell buffer

void renderGrid(RenderPipeline& rp, const float* cellColor, int fNumX, int fNumY, float h) {

}

// renderObstacle
// - pass center + radius as uniforms to obstacleShader
// - draw precomputed unit-circle VBO scaled at render time

void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius) {
 
}

// renderDestroy
// - glDeleteBuffers, glDeleteVertexArrays, glDeleteProgram for all objects

void renderDestroy(RenderPipeline& rp) {
    printf("[DUMMY] renderDestroy executed.\n");
}
