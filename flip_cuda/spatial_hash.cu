// flip_cuda/spatial_hash.cu
// NODE D (Nathanael) -- T2_pushApart

#include "flip_fluid.cuh"
#include "device_data.cuh"
#include <cub/cub.cuh>

// Pass 1: histogram_kernel
// - 1 thread per particle
// - atomicAdd(&numCellParticles[xi * pNumY + yi], 1)
// - zero numCellParticles before launch

// Pass 2: exclusive prefix scan
// - cub::DeviceScan::ExclusiveSum(numCellParticles, firstCellParticle, pNumCells)
// - allocate CUB temp buffer once at scene setup

// Pass 3: scatter_kernel
// - 1 thread per particle
// - slot = atomicAdd(&firstCellParticle[cellNr], 1)
// - cellParticleIds[slot] = i
// - note: run scan on a copy of firstCellParticle, scatter increments the copy

// Pass 4: separate_kernel (run numIters times)
// - 1 thread per particle
// - check 3x3 neighbor buckets via firstCellParticle
// - if dist(i, j) < 2 * particleRadius: push apart via atomicAdd on pos
// - diffuse colorR/G/B between overlapping pairs

// launchPushParticlesApart(DeviceData& d, int numParticles, int numIters,
//                          void* cub_temp, size_t cub_temp_bytes)
