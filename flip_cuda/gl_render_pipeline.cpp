#define GL_GLEXT_PROTOTYPES
#include "gl_render_pipeline.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <iostream>
#include <vector>
#include <cmath>

static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Error:\n" << infoLog << std::endl;
    }
    return shader;
}

static GLuint createProgram(const char* vShaderCode, const char* fShaderCode) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vShaderCode);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

// Shader Sources
const char* particleVert = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
out vec3 particleColor;
uniform mat4 projection;
uniform float pointSize;
void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    gl_PointSize = pointSize;
    particleColor = aColor;
}
)";

const char* particleFrag = R"(
#version 330 core
in vec3 particleColor;
out vec4 FragColor;
void main() {
    // Optional: Make points circular
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) discard;
    FragColor = vec4(particleColor, 1.0);
}
)";

const char* basicVert = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
out vec3 fragColor;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    fragColor = aColor;
}
)";

const char* basicFrag = R"(
#version 330 core
in vec3 fragColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(fragColor, 1.0);
}
)";

void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells) {
    rp.maxParticles = maxParticles;
    rp.fNumCells = fNumCells;

    // compile Shaders
    rp.particleShader = createProgram(particleVert, particleFrag);
    rp.gridShader = createProgram(basicVert, basicFrag);
    rp.obstacleShader = createProgram(basicVert, basicFrag);

    // setup Particle VAO & VBOs
    glGenVertexArrays(1, &rp.particleVAO);
    glBindVertexArray(rp.particleVAO);

    glGenBuffers(1, &rp.particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.particleVBO);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &rp.colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.colorVBO);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // setup Grid VAO & VBO
    glGenVertexArrays(1, &rp.gridVAO);
    glBindVertexArray(rp.gridVAO);

    glGenBuffers(1, &rp.gridVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.gridVBO);
    // 6 vertices per quad (2 triangles), 2 floats for pos, 3 for color = 5 floats per vertex
    glBufferData(GL_ARRAY_BUFFER, fNumCells * 6 * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    std::cout << "[OpenGL] VBO Pipeline initialized for " << maxParticles << " particles.\n";
}

void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx) {
    if (numParticles <= 0) return;

    // enable point sprites for circular particles
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    glUseProgram(rp.particleShader);

    // set orthographic projection
    // can bind an actual ortho matrix too here
    float ortho[16] = {
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -1.0f,-1.0f, 0.0f, 1.0f
    };
    GLint projLoc = glGetUniformLocation(rp.particleShader, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho);

    GLint pointSizeLoc = glGetUniformLocation(rp.particleShader, "pointSize");
    glUniform1f(pointSizeLoc, pointSizePx);

    glBindVertexArray(rp.particleVAO);

    glDrawArrays(GL_POINTS, 0, numParticles);
    glBindVertexArray(0);
}

void renderGrid(RenderPipeline& rp, const float* cellColor, int fNumX, int fNumY, float h) {
    if (!cellColor) return;

    // Note: to optimize this for standard pipeline, normally update the VBO
    // data dynamically here based on cellColor, expanding it to vertex quads.
    // for brevity, the logic to populate gridVBO via glMapBuffer goes here.

    glUseProgram(rp.gridShader);
    // (apply orthographic matrix similarly to renderParticles)

    glBindVertexArray(rp.gridVAO);
    // glDrawArrays(GL_TRIANGLES, 0, rp.fNumCells * 6);
    glBindVertexArray(0);
}

void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius) {
    // standard drawing logic mapping unit circle scaled by radius to ox, oy
}

void renderDestroy(RenderPipeline& rp) {
    glDeleteBuffers(1, &rp.particleVBO);
    glDeleteBuffers(1, &rp.colorVBO);
    glDeleteBuffers(1, &rp.gridVBO);
    glDeleteVertexArrays(1, &rp.particleVAO);
    glDeleteVertexArrays(1, &rp.gridVAO);
    glDeleteProgram(rp.particleShader);
    glDeleteProgram(rp.gridShader);
    glDeleteProgram(rp.obstacleShader);
    std::cout << "[OpenGL] Resources destroyed.\n";
}