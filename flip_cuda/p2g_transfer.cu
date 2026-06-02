// flip_cuda/p2g_transfer.cu
// NODE E (Brian) -- T4_p2g + T5_density

#include "flip_fluid.cuh"
#include "device_data.cuh"

// savePrevVelocities_kernel
// - 1 thread per cell
// - prevU[i]=u[i], prevV[i]=v[i], du[i]=dv[i]=u[i]=v[i]=0

// classifyCells_kernel (two passes)
// - pass A: 1 thread per cell -- cellType = s[i]==0 ? SOLID : AIR
// - pass B: 1 thread per particle -- atomicCAS(&cellType[cellNr], AIR_CELL, FLUID_CELL)

// p2gScatter_kernel (called twice: component 0=u, 1=v)
// - 1 thread per particle
// - bilinear weights to 4 surrounding face nodes (offset by 0 or h/2 per component)
// - atomicAdd(&fld[nr], pv * weight)
// - atomicAdd(&fldD[nr], weight)

// p2gNormalize_kernel (called twice: u then v)
// - 1 thread per cell
// - if du[i] > 0: u[i] /= du[i]

// restoreSolidCells_kernel
// - 1 thread per cell (2D: fNumX x fNumY)
// - if cell or left neighbor is SOLID: u[i*n+j] = prevU[i*n+j]
// - if cell or bottom neighbor is SOLID: v[i*n+j] = prevV[i*n+j]

// updateParticleDensity_kernel
// - 1 thread per particle
// - bilinear scatter to cell centers (both axes offset h/2)
// - atomicAdd(&particleDensity[nr], weight)
// - zero particleDensity before launch

// launchSavePrevVelocities(DeviceData& d, int fNumCells)
// launchClassifyCells(DeviceData& d, int fNumCells, int numParticles)
// launchP2G(DeviceData& d, int numParticles)
// launchRestoreSolidCells(DeviceData& d, int fNumCells)
// launchUpdateParticleDensity(DeviceData& d, int numParticles, int fNumCells)
