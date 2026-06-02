// flip_cuda/pressure_solver.cu
// NODE F (Fachriza) -- T6_pressure

#include "flip_fluid.cuh"
#include "device_data.cuh"

// pressureSolve_kernel(int parity)
// - parity 0 = RED cells (i+j) % 2 == 0
// - parity 1 = BLACK cells (i+j) % 2 == 1
// - 1 thread per cell, 2D grid (fNumX x fNumY)
// - skip if wrong parity, not FLUID_CELL, or out of bounds
// - div = u[right] - u[center] + v[top] - v[center]
// - if compensateDrift: div -= k * max(0, particleDensity[center] - restDensity)
// - pVal = -(div / sSum) * overRelaxation
// - update p[center], u/v faces of center and 4 neighbors
// - called alternately: RED pass then BLACK pass, for numIters/2 rounds

// launchRedBlackSolver(DeviceData& d, int numIters, float dt,
//                      float overRelaxation, bool compensateDrift,
//                      float restDensity)
