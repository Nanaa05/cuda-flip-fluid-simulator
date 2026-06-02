#define GL_GLEXT_PROTOTYPES
#include "gl_render_pipeline.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(sh, 512, nullptr, log);
        std::fprintf(stderr, "[GL shader] %s\n", log);
    }
    return sh;
}

static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

static const char* kParticleVert = R"(
#version 330 core
layout(location=0) in float aPosX;
layout(location=1) in float aPosY;
layout(location=2) in float aColorR;
layout(location=3) in float aColorG;
layout(location=4) in float aColorB;
uniform mat4 projection;
uniform float pointSize;
out vec3 fragColor;
void main() {
    gl_Position = projection * vec4(aPosX, aPosY, 0.0, 1.0);
    gl_PointSize = pointSize;
    fragColor = vec3(aColorR, aColorG, aColorB);
}
)";

static const char* kParticleFrag = R"(
#version 330 core
in vec3 fragColor;
out vec4 FragColor;
void main() {
    vec2 c = 2.0 * gl_PointCoord - 1.0;
    if (dot(c, c) > 1.0) discard;
    FragColor = vec4(fragColor, 1.0);
}
)";

static const char* kGridVert = R"(
#version 330 core
layout(location=0) in vec2 aQuadPos;
layout(location=1) in vec3 aColor;
uniform mat4 projection;
uniform float h;
uniform int fNumY;
out vec3 fragColor;
void main() {
    int ix = gl_InstanceID / fNumY;
    int iy = gl_InstanceID % fNumY;
    vec2 world = (aQuadPos + vec2(float(ix), float(iy))) * h;
    gl_Position = projection * vec4(world, 0.0, 1.0);
    fragColor = aColor;
}
)";

static const char* kGridFrag = R"(
#version 330 core
in vec3 fragColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(fragColor, 1.0);
}
)";

static const char* kObstacleVert = R"(
#version 330 core
layout(location=0) in vec2 aPos;
uniform mat4 projection;
uniform vec2 center;
uniform float radius;
void main() {
    vec2 world = center + aPos * radius;
    gl_Position = projection * vec4(world, 0.0, 1.0);
}
)";

static const char* kObstacleFrag = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
)";

static void buildOrtho(float* m, float w, float h) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = 2.0f / w;
    m[5] = 2.0f / h;
    m[10] = -1.0f;
    m[12] = -1.0f;
    m[13] = -1.0f;
    m[15] = 1.0f;
}

// === renderInit ===
void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells,
                int fNumX, int fNumY, float h) {
    rp.maxParticles = maxParticles;
    rp.fNumCells = fNumCells;
    rp.simWidth = fNumX * h;
    rp.simHeight = fNumY * h;
    buildOrtho(rp.projMatrix, rp.simWidth, rp.simHeight);

    rp.particleShader = makeProgram(kParticleVert, kParticleFrag);
    rp.gridShader = makeProgram(kGridVert, kGridFrag);
    rp.obstacleShader = makeProgram(kObstacleVert, kObstacleFrag);

    glGenVertexArrays(1, &rp.particleVAO);
    glBindVertexArray(rp.particleVAO);

    glGenBuffers(1, &rp.particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.particleVBO);
    glBufferData(GL_ARRAY_BUFFER, 2 * maxParticles * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float),
                          (void*)((size_t)maxParticles * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &rp.colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.colorVBO);
    glBufferData(GL_ARRAY_BUFFER, 3 * maxParticles * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float),
                          (void*)((size_t)maxParticles * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float),
                          (void*)((size_t)2 * maxParticles * sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindVertexArray(0);

    static const float quadVerts[12] = {
        0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f
    };

    glGenVertexArrays(1, &rp.gridVAO);
    glBindVertexArray(rp.gridVAO);

    glGenBuffers(1, &rp.gridQuadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.gridQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 0);

    glGenBuffers(1, &rp.gridColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.gridColorVBO);
    glBufferData(GL_ARRAY_BUFFER, fNumCells * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glBindVertexArray(0);

    const int N = 64;
    rp.obstacleVertCount = N + 2;
    std::vector<float> circle;
    circle.push_back(0.0f);
    circle.push_back(0.0f);
    for (int i = 0; i <= N; ++i) {
        float a = (float)i / N * 2.0f * 3.14159265f;
        circle.push_back(cosf(a));
        circle.push_back(sinf(a));
    }

    glGenVertexArrays(1, &rp.obstacleVAO);
    glBindVertexArray(rp.obstacleVAO);

    glGenBuffers(1, &rp.obstacleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.obstacleVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)circle.size() * sizeof(float), circle.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

// === renderParticles ===
void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx) {
    if (numParticles <= 0) return;
    glEnable(GL_PROGRAM_POINT_SIZE);
    glUseProgram(rp.particleShader);
    glUniformMatrix4fv(glGetUniformLocation(rp.particleShader, "projection"),
                       1, GL_FALSE, rp.projMatrix);
    glUniform1f(glGetUniformLocation(rp.particleShader, "pointSize"), pointSizePx);
    glBindVertexArray(rp.particleVAO);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glBindVertexArray(0);
}

// === renderGrid ===
void renderGrid(RenderPipeline& rp, const float* cellColor, int fNumX, int fNumY, float h) {
    if (!cellColor) return;
    int cells = fNumX * fNumY;
    glBindBuffer(GL_ARRAY_BUFFER, rp.gridColorVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, cells * 3 * sizeof(float), cellColor);
    glUseProgram(rp.gridShader);
    glUniformMatrix4fv(glGetUniformLocation(rp.gridShader, "projection"),
                       1, GL_FALSE, rp.projMatrix);
    glUniform1f(glGetUniformLocation(rp.gridShader, "h"), h);
    glUniform1i(glGetUniformLocation(rp.gridShader, "fNumY"), fNumY);
    glBindVertexArray(rp.gridVAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, cells);
    glBindVertexArray(0);
}

// === renderObstacle ===
void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius) {
    if (radius <= 0.0f) return;
    glUseProgram(rp.obstacleShader);
    glUniformMatrix4fv(glGetUniformLocation(rp.obstacleShader, "projection"),
                       1, GL_FALSE, rp.projMatrix);
    glUniform2f(glGetUniformLocation(rp.obstacleShader, "center"), ox, oy);
    glUniform1f(glGetUniformLocation(rp.obstacleShader, "radius"), radius);
    glBindVertexArray(rp.obstacleVAO);
    glDrawArrays(GL_TRIANGLE_FAN, 0, rp.obstacleVertCount);
    glBindVertexArray(0);
}

// === renderDestroy ===
void renderDestroy(RenderPipeline& rp) {
    glDeleteBuffers(1, &rp.particleVBO);
    glDeleteBuffers(1, &rp.colorVBO);
    glDeleteBuffers(1, &rp.gridQuadVBO);
    glDeleteBuffers(1, &rp.gridColorVBO);
    glDeleteBuffers(1, &rp.obstacleVBO);
    glDeleteVertexArrays(1, &rp.particleVAO);
    glDeleteVertexArrays(1, &rp.gridVAO);
    glDeleteVertexArrays(1, &rp.obstacleVAO);
    glDeleteProgram(rp.particleShader);
    glDeleteProgram(rp.gridShader);
    glDeleteProgram(rp.obstacleShader);
}
