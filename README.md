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

Proves the CUDA port computes the **same physics** as the CPU reference (not just faster). It runs both on an identical scene and compares each particle's **position** and **velocity** per frame.

```bash
make validate
./run_validation.sh
```

Or run a single test manually:

```bash
./flip_cuda/validate --lockstep
./flip_cuda/validate --lockstep --iters 500
./flip_cuda/validate --lockstep --no-separate
./flip_cuda/validate --lockstep --no-obstacle
```

`--lockstep` re-syncs GPU to the CPU state every frame so each frame measures one step from an identical start. Output shows per-frame **avg** and **worst** particle error (raw + % of cell size). A steady avg of a few percent of one cell = CPU and GPU agree.

## Project Structure

```
flip_cpu/    CPU baseline (single-thread C++)
flip_cuda/   CUDA port
```
