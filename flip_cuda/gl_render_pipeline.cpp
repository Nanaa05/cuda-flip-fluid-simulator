#define GL_GLEXT_PROTOTYPES
#include "gl_render_pipeline.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <iostream>
#include <vector>
#include <cmath>

// Helper untuk mengompilasi shader individu
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

// Helper untuk membuat shader program lengkap
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

// Shader Source untuk Partikel (SoA Attribute Map menggunakan input skalar)
const char* particleVert = R"(
#version 330 core
layout (location = 0) in float aPosX;
layout (location = 1) in float aPosY;
layout (location = 2) in float aColorR;
layout (location = 3) in float aColorG;
layout (location = 4) in float aColorB;

out vec3 particleColor;
uniform mat4 projection;
uniform float pointSize;

void main() {
    // Menyusun posisi koordinat dari attribute X dan Y terpisah di VRAM
    gl_Position = projection * vec4(aPosX, aPosY, 0.0, 1.0);
    gl_PointSize = pointSize;
    particleColor = vec3(aColorR, aColorG, aColorB);
}
)";

const char* particleFrag = R"(
#version 330 core
in vec3 particleColor;
out vec4 FragColor;
void main() {
    // Membentuk titik segiempat default agar bulat sempurna seperti lingkaran sprite
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) discard;
    FragColor = vec4(particleColor, 1.0);
}
)";

// Shader Source untuk Rendering Grid dan Obstacle
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

void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells, int fNumX, int fNumY, float h) {
    rp.maxParticles = maxParticles;
    rp.fNumCells = fNumCells;

    // Kompilasi shader program untuk masing-masing bagian render
    rp.particleShader = createProgram(particleVert, particleFrag);
    rp.gridShader = createProgram(basicVert, basicFrag);
    rp.obstacleShader = createProgram(basicVert, basicFrag);

    // =================================================================
    // 1. Inisialisasi VAO & VBO Partikel (Menggunakan Layout 2 VBO)
    // =================================================================
    glGenVertexArrays(1, &rp.particleVAO);
    glBindVertexArray(rp.particleVAO);

    // Alokasi particleVBO (Koordinat X, dilanjutkan koordinat Y di memori terpadu)
    glGenBuffers(1, &rp.particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.particleVBO);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Atribut 0: Posisi X (Mulai dari byte awal / offset 0)
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Atribut 1: Posisi Y (Mulai dari offset setengah buffer / setelah data X selesai)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(maxParticles * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Alokasi colorVBO (Komponen warna R, dilanjutkan G, dilanjutkan B di memori terpadu)
    glGenBuffers(1, &rp.colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.colorVBO);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Atribut 2: Red (Mulai dari offset 0)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    // Atribut 3: Green (Mulai dari offset sepertiga buffer / setelah data Red selesai)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(maxParticles * sizeof(float)));
    glEnableVertexAttribArray(3);

    // Atribut 4: Blue (Mulai dari offset dua pertiga buffer / setelah data Green selesai)
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(2 * maxParticles * sizeof(float)));
    glEnableVertexAttribArray(4);

    // =================================================================
    // 2. Inisialisasi VAO & VBO Grid
    // =================================================================
    glGenVertexArrays(1, &rp.gridVAO);
    glBindVertexArray(rp.gridVAO);

    glGenBuffers(1, &rp.gridQuadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.gridQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, fNumCells * 6 * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Atribut Grid: Posisi (Location 0) & Warna (Location 1)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // =================================================================
    // 3. Inisialisasi VAO & VBO Obstacle (Triangle Fan)
    // =================================================================
    glGenVertexArrays(1, &rp.obstacleVAO);
    glBindVertexArray(rp.obstacleVAO);

    glGenBuffers(1, &rp.obstacleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rp.obstacleVBO);

    // 32 segmen untuk lingkaran mulus + 1 pusat + 1 penutup = 34 vertices
    rp.obstacleVertCount = 34;
    glBufferData(GL_ARRAY_BUFFER, rp.obstacleVertCount * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Atribut Obstacle: Posisi (Location 0) & Warna (Location 1)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    std::cout << "[OpenGL] VBO 2-Resource Pipeline berhasil diinisialisasi.\n";
}

void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx) {
    if (numParticles <= 0) return;

    // Aktifkan point sprite untuk rendering lingkaran halus pada titik GL_POINTS
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    glUseProgram(rp.particleShader);

    // Orthographic Matrix bawaan untuk memetakan koordinat simulasi ke screen space
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
    
    // Bentuk geometri grid ke quads secara dinamis menggunakan mapping VBO langsung di GPU
    glBindBuffer(GL_ARRAY_BUFFER, rp.gridQuadVBO);
    float* ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        int idx = 0;
        for (int i = 0; i < fNumX; ++i) {
            for (int j = 0; j < fNumY; ++j) {
                int cellNr = i * fNumY + j;
                float r = cellColor[3 * cellNr + 0];
                float g = cellColor[3 * cellNr + 1];
                float b = cellColor[3 * cellNr + 2];

                // Jika sel kosong (udara/warna hitam), abaikan pengisian data render
                if (r == 0.0f && g == 0.0f && b == 0.0f) {
                    for (int k = 0; k < 30; ++k) ptr[idx++] = 0.0f;
                    continue;
                }

                float x0 = i * h;
                float y0 = j * h;
                float x1 = x0 + h;
                float y1 = y0 + h;

                // Segitiga 1
                ptr[idx++] = x0; ptr[idx++] = y0; ptr[idx++] = r; ptr[idx++] = g; ptr[idx++] = b;
                ptr[idx++] = x1; ptr[idx++] = y0; ptr[idx++] = r; ptr[idx++] = g; ptr[idx++] = b;
                ptr[idx++] = x0; ptr[idx++] = y1; ptr[idx++] = r; ptr[idx++] = g; ptr[idx++] = b;

                // Segitiga 2
                ptr[idx++] = x1; ptr[idx++] = y0; ptr[idx++] = r; ptr[idx++] = g; ptr[idx++] = b;
                ptr[idx++] = x1; ptr[idx++] = y1; ptr[idx++] = r; ptr[idx++] = g; ptr[idx++] = b;
                ptr[idx++] = x0; ptr[idx++] = y1; ptr[idx++] = r; ptr[idx++] = g; ptr[idx++] = b;
            }
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    // Aktifkan program shader grid
    glUseProgram(rp.gridShader);
    
    float ortho[16] = {
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -1.0f,-1.0f, 0.0f, 1.0f
    };
    GLint projLoc = glGetUniformLocation(rp.gridShader, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho);

    glBindVertexArray(rp.gridVAO);
    glDrawArrays(GL_TRIANGLES, 0, rp.fNumCells * 6);
    glBindVertexArray(0);
}

void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius) {
    glBindBuffer(GL_ARRAY_BUFFER, rp.obstacleVBO);
    float* ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        int idx = 0;
        float r_color = 0.85f, g_color = 0.35f, b_color = 0.35f; // Merah bata (Obstacle)

        // Titik pusat Triangle Fan
        ptr[idx++] = ox; ptr[idx++] = oy;
        ptr[idx++] = r_color; ptr[idx++] = g_color; ptr[idx++] = b_color;

        // Titik sekeliling lingkaran
        int numSegments = rp.obstacleVertCount - 2;
        for (int i = 0; i <= numSegments; ++i) {
            float theta = 2.0f * 3.1415926535f * float(i) / float(numSegments);
            ptr[idx++] = ox + radius * cosf(theta);
            ptr[idx++] = oy + radius * sinf(theta);
            ptr[idx++] = r_color; ptr[idx++] = g_color; ptr[idx++] = b_color;
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    glUseProgram(rp.obstacleShader);

    float ortho[16] = {
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -1.0f,-1.0f, 0.0f, 1.0f
    };
    GLint projLoc = glGetUniformLocation(rp.obstacleShader, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho);

    glBindVertexArray(rp.obstacleVAO);
    glDrawArrays(GL_TRIANGLE_FAN, 0, rp.obstacleVertCount);
    glBindVertexArray(0);
}

void renderDestroy(RenderPipeline& rp) {
    glDeleteBuffers(1, &rp.particleVBO);
    glDeleteBuffers(1, &rp.colorVBO);
    glDeleteBuffers(1, &rp.gridQuadVBO);
    glDeleteBuffers(1, &rp.obstacleVBO);
    glDeleteVertexArrays(1, &rp.particleVAO);
    glDeleteVertexArrays(1, &rp.gridVAO);
    glDeleteVertexArrays(1, &rp.obstacleVAO);
    glDeleteProgram(rp.particleShader);
    glDeleteProgram(rp.gridShader);
    glDeleteProgram(rp.obstacleShader);
    std::cout << "[OpenGL] Resource grafis 2 VBO berhasil dibersihkan dari memori.\n";
}