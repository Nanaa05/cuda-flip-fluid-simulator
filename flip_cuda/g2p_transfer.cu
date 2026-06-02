// flip_cuda/g2p_transfer.cu
// NODE G (Brian) -- T7_g2p

#include "flip_fluid.cuh"
#include "device_data.cuh"

// g2pGather_kernel (called twice: component 0=velX, 1=velY)
// - 1 thread per particle, 1D grid
// - bilinear gather from 4 surrounding face nodes (same stencil offsets as P2G)
// - validity flag per neighbor: 1 if cell or adjacent cell is not AIR
// - picV  = weighted average of current u/v values
// - corr  = weighted average of (u - prevU) deltas
// - flipV = particle_old_vel + corr
// - result = (1 - flipRatio) * picV + flipRatio * flipV
// - no atomics needed, each thread writes only to its own velX/Y[i]

// launchG2P(DeviceData& d, int numParticles, float flipRatio)
