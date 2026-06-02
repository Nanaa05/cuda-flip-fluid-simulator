# cuda-flip-fluid-simulator

FLIP fluid simulator ported from CPU (C++17) to GPU (CUDA). Both versions run side-by-side for benchmarking and numerical validation.

## Requirements

**CPU:** `g++`, `libGL`, `libX11`

**CUDA:** `nvcc`, CUDA toolkit, `libGL`, `libX11`

**Analysis:** `python3`, `matplotlib`, `numpy`

## Build

```bash
make cpu
make cuda
make validate
```

GPU arch defaults to `native`. Override if needed:

```bash
make cuda SM=sm_86
make cuda SM=sm_89
make cuda SM=sm_75
```

## Run

```bash
flip_cpu/flip
flip_cuda/flip
flip_cuda/flip --no-interop
```

`--no-interop` skips CUDA-OpenGL interop and uses plain `cudaMemcpy` for rendering. T10 measures the memcpy overhead instead of map/unmap, enabling direct interop cost comparison.

## Controls

| Key | Action |
|-----|--------|
| LMB drag | move obstacle |
| Space / P | pause / resume |
| G | toggle grid |
| R | reset scene |
| Q / Esc | quit |

Panel sliders: PIC/FLIP ratio, grid resolution. Checkboxes: gravity, separate particles, compensate drift, show grid/particles.

## Benchmark

Runs CPU, CUDA (with interop), and CUDA (without interop) across 4 resolutions (50, 100, 150, 200). Warmup 60 frames, measurement 600 frames.

```bash
./run_benchmark.sh
```

Output saved to `output/benchmark_results.log`. Each result line is labeled `[BENCHMARK_CPU_RESULT]`, `[BENCHMARK_CUDA_RESULT]`, or `[BENCHMARK_CUDA_NOINTEROP_RESULT]` with T1-T10 + T_total per stage. A `[T10_OVERHEAD]` line follows each CUDA result showing interop/memcpy cost as a percentage of T_total.

## Analysis

Parse benchmark log and generate speedup charts:

```bash
python3 analyze_results.py
```

Output saved to `output/benchmark_analysis.png`. Includes T_total vs resolution (log-scale) and speedup per pipeline stage at res=200.

## Numerical Validation

Proves the CUDA port computes the same physics as the CPU reference. Runs both on an identical scene and compares each particle's position and velocity per frame.

```bash
make validate
./run_validation.sh
```

Output saved to `output/validation_results.log`. Uses `--lockstep` mode: GPU is re-synced to CPU state each frame so each frame measures a single step from an identical start. A steady avg error of a few percent of one cell = CPU and GPU agree.

Manual flags:

| Flag | Description |
|------|-------------|
| `--lockstep` | fair per-frame comparison |
| `--iters N` | override pressure iterations |
| `--no-gravity` | disable gravity |
| `--no-obstacle` | disable obstacle |
| `--no-separate` | disable push-apart |
| `--res N` | grid resolution (default 100) |
| `--frames N` | frames to compare (default 120) |

## Project Structure

```
flip_cpu/           CPU baseline (single-thread C++)
flip_cuda/          CUDA port
output/             benchmark logs, validation logs, analysis charts
run_benchmark.sh    benchmark all resolutions
run_validation.sh   numerical validation
analyze_results.py  parse logs and generate charts
```
