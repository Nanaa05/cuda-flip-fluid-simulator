// flip_cuda/colors_reduce.cu
// T8 (Brian) -- updateParticleColors + updateCellColors + computeRestDensity

#include "flip_fluid.cuh"
#include "device_data.cuh"

// computeRestDensity_kernel
// - parallel reduction over FLUID_CELL entries of particleDensity
// - shared memory reduction per block, partial sums written to d_partialSums
// - host reads back and divides sum by count, stores to d_restDensity
// - alternative: cub::DeviceReduce with conditional functor (skip non-fluid)

// updateParticleColors_kernel
// - 1 thread per particle
// - drift R and G down by 0.01, B up by 0.01, clamp to [0,1]
// - if particleDensity[cellNr] / restDensity < 0.7: set R=G=0.8, B=1.0

// updateCellColors_kernel
// - 1 thread per cell
// - SOLID: cellColor[3*i..] = 0.5
// - FLUID: setSciColor(density / restDensity, 0, 2) -- rainbow blue to red
// - AIR:   cellColor[3*i..] = 0.0
// - __device__ setSciColor inlined in this file

// launchComputeRestDensity(DeviceData& d, int fNumCells) -> float
// launchUpdateParticleColors(DeviceData& d, int numParticles)
// launchUpdateCellColors(DeviceData& d, int fNumCells)
