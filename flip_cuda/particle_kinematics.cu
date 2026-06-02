// flip_cuda/particle_kinematics.cu
// NODE C (Nathanael) -- T1_integrate + T3_collisions

#include "flip_fluid.cuh"
#include "device_data.cuh"

// integrateParticles_kernel
// - 1 thread per particle, 1D grid
// - velY[i] += gravity * dt
// - posX[i] += velX[i] * dt
// - posY[i] += velY[i] * dt

// handleCollisions_kernel
// - 1 thread per particle, 1D grid
// - if dist(pos, obstacle) < obstacleRadius + particleRadius: set vel to obstacleVel
// - clamp pos to wall bounds [minX..maxX, minY..maxY], zero vel on contact

// launchIntegrateParticles(DeviceData& d, int numParticles, float dt, float gravity)
// launchHandleCollisions(DeviceData& d, int numParticles,
//                        float obstacleX, float obstacleY, float obstacleRadius,
//                        float obstacleVelX, float obstacleVelY)
