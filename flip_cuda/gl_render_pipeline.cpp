// flip_cuda/gl_render_pipeline.cpp
// NODE I1 (Fachriza) -- replaces legacy glBegin/glEnd with VBO pipeline

#include "gl_render_pipeline.h"
#include <GL/gl.h>
#include <cstdio>

// renderInit
// - glGenVertexArrays + glGenBuffers for particle and grid VAO/VBO
// - glBufferData with GL_DYNAMIC_DRAW for particleVBO, colorVBO, gridVBO
// - compile vertex + fragment shaders for particles, grid cells, obstacle

// renderParticles
// - glBindVertexArray(particleVAO)
// - glUseProgram(particleShader), set pointSize uniform
// - glDrawArrays(GL_POINTS, 0, numParticles)

// renderGrid
// - glBufferSubData cellColor into gridVBO
// - draw: either instanced quads or pre-expanded 4-vertex-per-cell buffer

// renderObstacle
// - pass center + radius as uniforms to obstacleShader
// - draw precomputed unit-circle VBO scaled at render time

// renderDestroy
// - glDeleteBuffers, glDeleteVertexArrays, glDeleteProgram for all objects
