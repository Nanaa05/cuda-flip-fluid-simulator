// flip_cuda/main.cu
// Entry point for the CUDA binary.

#include "flip_fluid.cuh"
#include "device_data.cuh"
#include "gl_render_pipeline.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <cstdio>

// startup sequence:
// - parse --no-vsync, --res N
// - createWindow() X11 + GLX (same as flip_cpu)
// - setupScene(): seed particles on host, call gpuSimulateInit()
// - renderInit(), then interopInit()

// per-frame loop:
// - pump X11 events (keyboard/mouse, same logic as flip_cpu)
// - interopMapResources()
// - if obstacle moved: launchCarveObstacle()
// - if not paused: gpuSimulate()
// - interopUnmapResources()
// - T9: renderParticles, renderGrid, renderObstacle, UI overlay, glXSwapBuffers
// - NODE J: accumulate cudaEvent timings, print every 60 frames

// scene reset (R key or resolution change):
// - interopDestroy, renderDestroy, DeviceData::free
// - re-run gpuSimulateInit, renderInit, interopInit

// NODE J print format (stdout, every 60 frames):
// [GPU] frame=60 res=100 T1=0.02ms T2=0.31ms T3=0.01ms T4=0.18ms
//       T5=0.09ms T6=1.2ms T7=0.11ms T8=0.05ms T9=0.4ms T10=0.03ms
//       T_total=2.4ms particles=4096
