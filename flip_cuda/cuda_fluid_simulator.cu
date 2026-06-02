// flip_cuda/cuda_fluid_simulator.cu
// NODE H (Fachriza) -- master GPU pipeline, ties all kernels together

#include "flip_fluid.cuh"
#include "device_data.cuh"

// gpuSimulateInit(DeviceData& d, SimParams& params, const float* hPosX,
//                 const float* hPosY, const float* hS, int numParticles)
// - cudaMemcpyToSymbol(d_params, &params)
// - DeviceData::allocate()
// - cudaMemcpy initial posX/posY, s[] to device
// - zero vel, u, v, p, particleDensity on device
// - allocate CUB temp buffer for spatial hash scan (Node D)
// - call interopInit() if interop enabled

// gpuSimulate(DeviceData& d, float dt, float gravity, float flipRatio,
//             int numPressureIters, int numParticleIters,
//             float overRelaxation, bool compensateDrift,
//             bool separateParticles, float obstacleX, float obstacleY,
//             float obstacleRadius, float obstacleVelX, float obstacleVelY,
//             int numSubSteps)
// pipeline per substep:
//   T1: launchIntegrateParticles
//   T2: launchPushParticlesApart  (if separateParticles)
//   T3: launchHandleCollisions
//   T4: launchSavePrevVelocities, launchClassifyCells, launchP2G, launchRestoreSolidCells
//   T5: launchUpdateParticleDensity
//       launchComputeRestDensity  (first frame only, sets particleRestDensity)
//   T6: launchRedBlackSolver
//   T7: launchG2P
// after substeps:
//   T8: launchComputeRestDensity, launchUpdateParticleColors, launchUpdateCellColors

// carveObstacle_kernel (optional, called before T1 when obstacle moves)
// - 1 thread per interior cell
// - if dist(cell_center, obstacle) < radius: s[i]=0, set u/v faces to obstacle vel
// - else: s[i]=1
