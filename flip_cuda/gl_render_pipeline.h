#pragma once
#include <GL/gl.h>

struct RenderPipeline {
    // VAO untuk partikel
    GLuint particleVAO = 0;
    
    // 2 VBO terpadu untuk koordinat dan warna (Node I2)
    GLuint particleVBO = 0;   // Menampung koordinat X dan Y (SoA concatenated)
    GLuint colorVBO    = 0;   // Menampung warna R, G, B (SoA concatenated)

    // Grid instanced rendering
    GLuint gridVAO      = 0;
    GLuint gridQuadVBO  = 0;   // Static unit quad
    GLuint gridColorVBO = 0;   // Warna per sel grid

    // Obstacle triangle fan
    GLuint obstacleVAO       = 0;
    GLuint obstacleVBO       = 0;
    int    obstacleVertCount = 0;

    GLuint particleShader = 0;
    GLuint gridShader     = 0;
    GLuint obstacleShader = 0;

    int   maxParticles = 0;
    int   fNumCells    = 0;
    float simWidth     = 0.0f;
    float simHeight    = 0.0f;
};

// Inisialisasi: Alokasi VAO, VBO (GL_DYNAMIC_DRAW), compile shader
void renderInit(RenderPipeline& rp, int maxParticles, int fNumCells,
                int fNumX, int fNumY, float h);

// Menggambar partikel langsung dari memori VBO tanpa pemindahan data ke host
void renderParticles(RenderPipeline& rp, int numParticles, float pointSizePx);

// Menggambar grid warna
void renderGrid(RenderPipeline& rp, const float* cellColor, int fNumX, int fNumY, float h);

// Menggambar objek penghalang (obstacle)
void renderObstacle(RenderPipeline& rp, float ox, float oy, float radius);

// Pembersihan memori OpenGL
void renderDestroy(RenderPipeline& rp);