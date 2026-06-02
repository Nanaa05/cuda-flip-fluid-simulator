#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <device_launch_parameters.h>

// === integrateParticles_kernel ===
__global__ void integrateParticles_kernel(
    float* posX, float* posY,
    float* velX, float* velY,
    float dt, float gravity, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        velY[i] += gravity * dt;
        posX[i] += velX[i] * dt;
        posY[i] += velY[i] * dt;
    }
}

// === handleCollisions_kernel ===
__global__ void handleCollisions_kernel(
    float* posX, float* posY,
    float* velX, float* velY,
    float obstacleX, float obstacleY, float obstacleRadius,
    float obstacleVelX, float obstacleVelY, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float x = posX[i];
    float y = posY[i];
    float vx = velX[i];
    float vy = velY[i];

    float hh = 1.0f / d_params.fInvSpacing;
    float r = d_params.particleRadius;
    float minDist = obstacleRadius + r;

    float minX = hh + r;
    float maxX = (d_params.fNumX - 1) * hh - r;
    float minY = hh + r;
    float maxY = (d_params.fNumY - 1) * hh - r;

    float dx = x - obstacleX;
    float dy = y - obstacleY;

    if (dx * dx + dy * dy < minDist * minDist) {
        vx = obstacleVelX;
        vy = obstacleVelY;
    }

    if (x < minX) { x = minX; vx = 0.0f; }
    if (x > maxX) { x = maxX; vx = 0.0f; }
    if (y < minY) { y = minY; vy = 0.0f; }
    if (y > maxY) { y = maxY; vy = 0.0f; }

    posX[i] = x;
    posY[i] = y;
    velX[i] = vx;
    velY[i] = vy;
}

// === launchIntegrateParticles ===
void launchIntegrateParticles(DeviceData& d, int numParticles, float dt, float gravity) {
    if (numParticles <= 0) return;
    int blocks = (numParticles + 255) / 256;
    integrateParticles_kernel<<<blocks, 256>>>(
        d.posX, d.posY, d.velX, d.velY, dt, gravity, numParticles
    );
}

// === launchHandleCollisions ===
void launchHandleCollisions(DeviceData& d, int numParticles,
                            float obstacleX, float obstacleY, float obstacleRadius,
                            float obstacleVelX, float obstacleVelY)
{
    if (numParticles <= 0) return;
    int blocks = (numParticles + 255) / 256;
    handleCollisions_kernel<<<blocks, 256>>>(
        d.posX, d.posY, d.velX, d.velY,
        obstacleX, obstacleY, obstacleRadius,
        obstacleVelX, obstacleVelY, numParticles
    );
}
