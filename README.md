# cuda-flip-fluid-simulator

FLIP fluid simulator ported from CPU (C++17) to GPU (CUDA). Both versions run side-by-side for benchmarking.

## Requirements

**CPU build:** `g++`, `libGL`, `libX11`

**CUDA build:** `nvcc`, CUDA toolkit, `libGL`, `libX11`

## Build & Run

```bash
make cpu        # build flip_cpu/flip
make cuda       # build flip_cuda/flip_cuda
make run-cpu    # build + run CPU
make run-cuda   # build + run CUDA
make clean      # clean both
```

GPU arch defaults to `native` (auto-detect). Override only if needed:

```bash
make cuda SM=sm_86   # RTX 30xx
make cuda SM=sm_89   # RTX 40xx
make cuda SM=sm_75   # GTX 16xx
```

Run directly after build:

```bash
flip_cpu/flip
flip_cuda/flip_cuda
```

## Controls

| Key | Action |
|-----|--------|
| LMB drag | move obstacle |
| Space / P | pause / resume |
| G | toggle grid |
| R | reset scene |
| Q / Esc | quit |

## Numerical Validation (CPU vs GPU)

Runs both CPU and GPU on identical scenes and compares particle positions/velocities per frame.

```bash
make validate          # build validator (requires CUDA)
./flip_cuda/validate   # run with defaults (res=100, 120 frames)
```

Options:

```bash
./flip_cuda/validate --lockstep      # per-step error (re-syncs GPU to CPU each frame) -- the real correctness metric
./flip_cuda/validate --iters 500     # override pressure iterations (more = better-converged, smaller error)
./flip_cuda/validate --res 100       # grid resolution (default 100)
./flip_cuda/validate --frames 200    # number of frames to compare (default 120)
./flip_cuda/validate --no-obstacle   # disable obstacle (isolates pure fluid physics)
./flip_cuda/validate --no-gravity    # disable gravity
./flip_cuda/validate --no-separate   # disable push-apart (isolates P2G/G2P only)
```

Output reports `max / mean / rms` position and velocity error per frame.

**Two modes:**

- **Free-run (default):** both sims run independently from the same seed. Because FLIP is chaotic, tiny per-step differences (parallel red-black vs sequential SOR pressure solve, parallel push-apart, non-associative atomic sums) double every few frames and accumulated error reaching domain scale is *expected and normal* -- not a bug.
- **Lockstep (`--lockstep`):** the GPU is re-synced to the CPU state every frame, so each frame measures a single step from an identical state. This is the meaningful correctness test: if the per-step error stays small and bounded, the GPU pipeline is correct. Raising `--iters` shrinks it further by converging the pressure solve.

## Project Structure

```
flip_cpu/    CPU baseline (single-thread C++)
flip_cuda/   CUDA port
```
