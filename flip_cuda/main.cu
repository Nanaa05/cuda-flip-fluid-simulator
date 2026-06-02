#define GL_GLEXT_PROTOTYPES
#include "flip_fluid.cuh"
#include "device_data.cuh"
#include "gl_render_pipeline.h"
#include "ui.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <cuda_gl_interop.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__constant__ SimParams d_params;

extern void gpuSimulate(DeviceData& d, int numParticles, float dt, float gravity,
                        float flipRatio, int numPressureIters, int numParticleIters,
                        float overRelaxation, bool compensateDrift, bool separateParticles,
                        float obstacleX, float obstacleY, float obstacleRadius,
                        float obstacleVelX, float obstacleVelY, int numSubSteps);
extern void gpuUpdateColors(DeviceData& d, int numParticles);

extern void interopInit(DeviceData& d, const RenderPipeline& rp);
extern void interopMapResources(DeviceData& d);
extern void interopUnmapResources(DeviceData& d);
extern void interopDestroy(DeviceData& d);

static constexpr int CANVAS_W = 900;
static constexpr int CANVAS_H = 700;
static constexpr float SIM_HEIGHT = 3.0f;
static constexpr float SIM_WIDTH = (float)CANVAS_W / ((float)CANVAS_H / SIM_HEIGHT);

struct Scene {
    float gravity = -9.81f;
    float dt = 1.0f / 60.0f;
    float flipRatio = 0.9f;
    int numPressureIters = 50;
    int numParticleIters = 2;
    int numSubSteps = 1;
    long frameNr = 0;
    float overRelaxation = 1.9f;
    bool compensateDrift = true;
    bool separateParticles = true;
    float obstacleX = 3.0f;
    float obstacleY = 2.0f;
    float obstacleRadius = 0.15f;
    float obstacleVelX = 0.0f;
    float obstacleVelY = 0.0f;
    bool paused = true;
    bool showGrid = false;
    bool showParticles = true;
    bool showObstacle = true;
    int resolution = 100;
} scene;

struct Telemetry {
    float t9 = 0.0f;
    float t10 = 0.0f;
    float t_total = 0.0f;
    void reset() { t9 = t10 = t_total = 0.0f; }
} telemetry;

struct AppWindow {
    Display* dpy = nullptr;
    Window xwin = 0;
    GLXContext glc = nullptr;
    int width = CANVAS_W;
    int height = CANVAS_H;
    bool running = true;
} w;

DeviceData d;
RenderPipeline rp;

std::vector<float> h_posX;
std::vector<float> h_posY;
std::vector<float> h_s;
std::vector<float> h_colorR;
std::vector<float> h_colorG;
std::vector<float> h_colorB;
int g_numParticles = 0;
int g_fNumX = 0;
int g_fNumY = 0;
float g_h = 0.0f;
float g_particleRadius = 0.0f;
int g_fNumCells = 0;

static int s_glxAttrs[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24, None };

static void createWindow() {
    w.dpy = XOpenDisplay(nullptr);
    if (!w.dpy) { std::fprintf(stderr, "Cannot open X display\n"); std::exit(1); }

    XVisualInfo* vi = glXChooseVisual(w.dpy, DefaultScreen(w.dpy), s_glxAttrs);
    if (!vi) { std::fprintf(stderr, "No suitable GLX visual\n"); std::exit(1); }

    Colormap cmap = XCreateColormap(w.dpy, RootWindow(w.dpy, vi->screen), vi->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask |
                     PointerMotionMask | StructureNotifyMask;

    w.xwin = XCreateWindow(w.dpy, RootWindow(w.dpy, vi->screen), 0, 0,
                           w.width, w.height, 0, vi->depth, InputOutput,
                           vi->visual, CWColormap | CWEventMask, &swa);
    XStoreName(w.dpy, w.xwin, "FLIP Fluid Simulator (CUDA)");
    XMapWindow(w.dpy, w.xwin);

    w.glc = glXCreateContext(w.dpy, vi, nullptr, GL_TRUE);
    if (!w.glc) { std::fprintf(stderr, "glXCreateContext failed\n"); std::exit(1); }
    glXMakeCurrent(w.dpy, w.xwin, w.glc);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

static void setupScene() {
    float h = SIM_HEIGHT / (float)scene.resolution;
    float r = 0.3f * h;
    g_particleRadius = r;
    float dx = 2.0f * r;
    float dy = std::sqrt(3.0f) / 2.0f * dx;

    float relWaterHeight = 0.8f;
    float relWaterWidth = 0.6f;

    int numX = (int)std::floor((relWaterWidth * SIM_WIDTH - 2.0f * h - 2.0f * r) / dx);
    int numY = (int)std::floor((relWaterHeight * SIM_HEIGHT - 2.0f * h - 2.0f * r) / dy);
    if (numX < 1) numX = 1;
    if (numY < 1) numY = 1;

    int maxParticles = numX * numY;

    g_fNumX = (int)std::floor(SIM_WIDTH / h) + 1;
    g_fNumY = (int)std::floor(SIM_HEIGHT / h) + 1;
    g_h = std::max(SIM_WIDTH / (float)g_fNumX, SIM_HEIGHT / (float)g_fNumY);
    float fInvSpacing = 1.0f / g_h;
    g_fNumCells = g_fNumX * g_fNumY;

    float pSpacing = 2.0f * r;
    float pInvSpacing = 1.0f / pSpacing;
    int pNumX = (int)std::floor(SIM_WIDTH / pSpacing) + 1;
    int pNumY = (int)std::floor(SIM_HEIGHT / pSpacing) + 1;
    int pNumCells = pNumX * pNumY;

    if (scene.resolution <= 100) scene.numSubSteps = 1;
    else if (scene.resolution <= 140) scene.numSubSteps = 2;
    else if (scene.resolution <= 180) scene.numSubSteps = 3;
    else scene.numSubSteps = 4;
    scene.numPressureIters = 50 + std::max(0, (scene.resolution - 100)) / 2;

    SimParams hostParams;
    hostParams.fNumX = g_fNumX;
    hostParams.fNumY = g_fNumY;
    hostParams.fNumCells = g_fNumCells;
    hostParams.h = g_h;
    hostParams.fInvSpacing = fInvSpacing;
    hostParams.pNumX = pNumX;
    hostParams.pNumY = pNumY;
    hostParams.pNumCells = pNumCells;
    hostParams.pInvSpacing = pInvSpacing;
    hostParams.particleRadius = r;
    hostParams.density = 1000.0f;
    hostParams.maxParticles = maxParticles;
    cudaMemcpyToSymbol(d_params, &hostParams, sizeof(SimParams));

    h_s.assign(g_fNumCells, 1.0f);
    for (int i = 0; i < g_fNumX; ++i) {
        for (int j = 0; j < g_fNumY; ++j) {
            if (i == 0 || i == g_fNumX - 1 || j == 0)
                h_s[i * g_fNumY + j] = 0.0f;
        }
    }

    h_posX.clear(); h_posY.clear();
    for (int i = 0; i < numX; ++i) {
        for (int j = 0; j < numY; ++j) {
            float offset = (j % 2 == 0) ? 0.0f : r;
            h_posX.push_back(g_h + r + dx * i + offset);
            h_posY.push_back(g_h + r + dy * j);
        }
    }
    g_numParticles = (int)h_posX.size();

    h_colorR.assign(g_numParticles, 0.0f);
    h_colorG.assign(g_numParticles, 0.0f);
    h_colorB.assign(g_numParticles, 1.0f);

    d.allocate(g_fNumCells, pNumCells, maxParticles);
    renderInit(rp, maxParticles, g_fNumCells, g_fNumX, g_fNumY, g_h);
    interopInit(d, rp);
    launchDeviceDataReset(d, h_posX.data(), h_posY.data(), h_s.data(), g_numParticles, g_fNumCells);
    
    // Deterministic Obstacle Setup
    scene.obstacleX = 3.0f;
    scene.obstacleY = 2.0f;
    scene.frameNr = 0;
}

static void cleanUp() {
    interopDestroy(d);
    renderDestroy(rp);
    d.free();
    if (w.glc) { glXMakeCurrent(w.dpy, None, nullptr); glXDestroyContext(w.dpy, w.glc); }
    if (w.xwin) XDestroyWindow(w.dpy, w.xwin);
    if (w.dpy) XCloseDisplay(w.dpy);
}

int main(int argc, char** argv) {
    bool isBenchmark = false;
    int benchmarkRes = 100;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            isBenchmark = true;
            benchmarkRes = std::atoi(argv[++i]);
        }
    }

    setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 0);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 0);

    createWindow();

    {
        unsigned int glDeviceCount = 0;
        int glDevice = 0;
        cudaError_t err = cudaGLGetDevices(&glDeviceCount, &glDevice, 1, cudaGLDeviceListCurrentFrame);
        if (err == cudaSuccess && glDeviceCount > 0)
            cudaSetDevice(glDevice);
    }

    if (isBenchmark) {
        scene.resolution = benchmarkRes;
    }
    setupScene();

    cudaEvent_t evFrameStart, evFrameStop;
    cudaEvent_t evMapStart, evMapStop;
    cudaEvent_t evUnmapStart, evUnmapStop;
    cudaEvent_t evRenderStart, evRenderStop;
    cudaEventCreate(&evFrameStart); cudaEventCreate(&evFrameStop);
    cudaEventCreate(&evMapStart);   cudaEventCreate(&evMapStop);
    cudaEventCreate(&evUnmapStart); cudaEventCreate(&evUnmapStop);
    cudaEventCreate(&evRenderStart); cudaEventCreate(&evRenderStop);

    bool mouseDown = false;
    bool mouseDownPrev = false;
    bool mousePressedEdge = false;
    bool mouseReleasedEdge = false;
    int mousePxX = 0, mousePxY = 0;
    float mouseSimX = 0.0f, mouseSimY = 0.0f;
    bool dragOwnedByUI = false;
    bool gravityOn = (scene.gravity != 0.0f);

    auto fpsT0 = std::chrono::steady_clock::now();
    int fpsFrames = 0;
    double lastFps = 0.0;

    glViewport(0, 0, w.width, w.height);

    while (w.running) {
        auto t_frame_start = std::chrono::steady_clock::now();
        cudaEventRecord(evFrameStart, 0);
        mousePressedEdge = false;
        mouseReleasedEdge = false;

        if (isBenchmark) {
            scene.paused = false;
        }

        while (XPending(w.dpy)) {
            XEvent e;
            XNextEvent(w.dpy, &e);
            if (e.type == ConfigureNotify) {
                w.width = e.xconfigure.width;
                w.height = e.xconfigure.height;
            } else if (e.type == KeyPress) {
                KeySym ks = XLookupKeysym(&e.xkey, 0);
                if (ks == XK_Escape || ks == XK_q) w.running = false;
                else if (ks == XK_space || ks == XK_p) scene.paused = !scene.paused;
                else if (ks == XK_g) scene.showGrid = !scene.showGrid;
                else if (ks == XK_r) {
                    interopDestroy(d); renderDestroy(rp); d.free(); setupScene();
                }
            } else if (e.type == ButtonPress && e.xbutton.button == Button1) {
                mouseDown = true; mousePressedEdge = true;
                mousePxX = e.xbutton.x; mousePxY = e.xbutton.y;
                mouseSimX = (float)e.xbutton.x / w.width * SIM_WIDTH;
                mouseSimY = (1.0f - (float)e.xbutton.y / w.height) * SIM_HEIGHT;
            } else if (e.type == ButtonRelease && e.xbutton.button == Button1) {
                mouseDown = false; mouseReleasedEdge = true;
                mousePxX = e.xbutton.x; mousePxY = e.xbutton.y;
            } else if (e.type == MotionNotify) {
                mousePxX = e.xmotion.x; mousePxY = e.xmotion.y;
                mouseSimX = (float)e.xmotion.x / w.width * SIM_WIDTH;
                mouseSimY = (1.0f - (float)e.xmotion.y / w.height) * SIM_HEIGHT;
            }
        }

        int uiMouseY = w.height - mousePxY;
        const int kPanelX = 10, kPanelY = 10, kPanelW = 160, kPanelH = 260;
        bool mouseOnPanel = (mousePxX >= kPanelX && mousePxX < kPanelX + kPanelW &&
                             uiMouseY >= kPanelY && uiMouseY < kPanelY + kPanelH);
        if (mousePressedEdge && mouseOnPanel) dragOwnedByUI = true;
        if (!mouseDown) dragOwnedByUI = false;

        if (mouseDown && !dragOwnedByUI) {
            if (!mouseDownPrev) {
                scene.obstacleX = mouseSimX; scene.obstacleY = mouseSimY;
                scene.obstacleVelX = 0.0f; scene.obstacleVelY = 0.0f;
                scene.paused = false;
            } else {
                scene.obstacleVelX = (mouseSimX - scene.obstacleX) / scene.dt;
                scene.obstacleVelY = (mouseSimY - scene.obstacleY) / scene.dt;
                scene.obstacleX = mouseSimX; scene.obstacleY = mouseSimY;
            }
            mouseDownPrev = true;
        } else {
            if (mouseDownPrev) { scene.obstacleVelX = 0.0f; scene.obstacleVelY = 0.0f; }
            mouseDownPrev = false;
        }

        float t10Ms = 0.0f;
        if (!scene.paused) {
            if (isBenchmark && scene.frameNr == 60) {
                g_gpu_telemetry.reset();
                telemetry.reset();
            }

            cudaEventRecord(evMapStart, 0);
            interopMapResources(d);
            cudaEventRecord(evMapStop, 0);

            gpuSimulate(d, g_numParticles, scene.dt, scene.gravity, scene.flipRatio,
                        scene.numPressureIters, scene.numParticleIters, scene.overRelaxation,
                        scene.compensateDrift, scene.separateParticles,
                        scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
                        scene.obstacleVelX, scene.obstacleVelY, scene.numSubSteps);


            gpuUpdateColors(d, g_numParticles);

            cudaMemcpy(h_posX.data(), d.posX, g_numParticles * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_posY.data(), d.posY, g_numParticles * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_colorR.data(), d.colorR, g_numParticles * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_colorG.data(), d.colorG, g_numParticles * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_colorB.data(), d.colorB, g_numParticles * sizeof(float), cudaMemcpyDeviceToHost);

            cudaEventRecord(evUnmapStart, 0);
            interopUnmapResources(d);
            cudaEventRecord(evUnmapStop, 0);
            cudaEventSynchronize(evUnmapStop);

            float mapMs = 0.0f, unmapMs = 0.0f;
            cudaEventElapsedTime(&mapMs, evMapStart, evMapStop);
            cudaEventElapsedTime(&unmapMs, evUnmapStart, evUnmapStop);
            t10Ms = mapMs + unmapMs;
        }

        cudaEventRecord(evRenderStart, 0);
        glViewport(0, 0, w.width, w.height);
        glClear(GL_COLOR_BUFFER_BIT);

        if (scene.showGrid) {
            std::vector<float> h_cellColor(g_fNumCells * 3);
            cudaMemcpy(h_cellColor.data(), d.cellColor, g_fNumCells * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            renderGrid(rp, h_cellColor.data(), g_fNumX, g_fNumY, g_h);
        }

        if (scene.showParticles && g_numParticles > 0) {
            glUseProgram(0); glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
            glOrtho(0.0, rp.simWidth, 0.0, rp.simHeight, -1.0, 1.0);
            glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
            float pxPerSimUnit = (float)w.height / SIM_HEIGHT;
            float diameterPx = 2.0f * g_particleRadius * pxPerSimUnit;
            if (diameterPx < 1.0f) diameterPx = 1.0f;
            glDisable(GL_PROGRAM_POINT_SIZE); glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
            glPointSize(diameterPx); glBegin(GL_POINTS);
            for (int i = 0; i < g_numParticles; ++i) {

                glColor3f(h_colorR[i], h_colorG[i], h_colorB[i]); 
                glVertex2f(h_posX[i], h_posY[i]);
            }
            glEnd(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glPopMatrix();
        }

        if (scene.showObstacle) renderObstacle(rp, scene.obstacleX, scene.obstacleY, scene.obstacleRadius);

        glUseProgram(0); flipcpu_ui::setProjectionToPixels(w.width, w.height);
        flipcpu_ui::Input uin;
        uin.screenW = w.width; uin.screenH = w.height; uin.mouseX = mousePxX; uin.mouseY = mousePxY;
        uin.mouseDown = mouseDown; uin.mousePressed = mousePressedEdge && mouseOnPanel; uin.mouseReleased = mouseReleasedEdge;
        flipcpu_ui::begin(uin); flipcpu_ui::beginPanel(kPanelX, kPanelY, kPanelW, kPanelH, "Controls");
        flipcpu_ui::text("FPS: %.1f", lastFps); flipcpu_ui::text("Particles: %d", g_numParticles); flipcpu_ui::text("Frame: %ld", scene.frameNr);
        flipcpu_ui::checkbox("Particles", &scene.showParticles); flipcpu_ui::checkbox("Grid", &scene.showGrid);
        flipcpu_ui::checkbox("Compensate Drift", &scene.compensateDrift); flipcpu_ui::checkbox("Separate Particles", &scene.separateParticles);
        if (flipcpu_ui::checkbox("Gravity", &gravityOn)) { scene.gravity = gravityOn ? -9.81f : 0.0f; }
        flipcpu_ui::slider("PIC <-> FLIP", &scene.flipRatio, 0.0f, 1.0f);
        float resFloat = (float)scene.resolution; flipcpu_ui::slider("Grid Res", &resFloat, 30.0f, 200.0f);
        int newRes = (int)(resFloat + 0.5f);
        if (newRes != scene.resolution) { 
            scene.resolution = newRes; interopDestroy(d); renderDestroy(rp); d.free(); setupScene(); 
        }
        flipcpu_ui::checkbox("Pause", &scene.paused); 
        if (flipcpu_ui::button("Reset")) { 
            interopDestroy(d); renderDestroy(rp); d.free(); setupScene(); 
        }
        flipcpu_ui::endPanel(); flipcpu_ui::restoreProjection();

        glXSwapBuffers(w.dpy, w.xwin);

        cudaEventRecord(evRenderStop, 0); cudaEventSynchronize(evRenderStop);
        float renderMs = 0.0f; cudaEventElapsedTime(&renderMs, evRenderStart, evRenderStop);
        cudaEventRecord(evFrameStop, 0); cudaEventSynchronize(evFrameStop);
        float totalMs = 0.0f; cudaEventElapsedTime(&totalMs, evFrameStart, evFrameStop);

        if (!scene.paused) {
            if (isBenchmark) {
                if (scene.frameNr == 660) {
                    float N = 600.0f;
                    std::printf("[BENCHMARK_CUDA_RESULT] res=%d T1=%.3fms T2=%.3fms T3=%.3fms T4=%.3fms T5=%.3fms T6=%.3fms T7=%.3fms T8=%.3fms T9=%.3fms T10=%.3fms T_total=%.3fms particles=%d\n",
                                scene.resolution, g_gpu_telemetry.t1 / N, g_gpu_telemetry.t2 / N, g_gpu_telemetry.t3 / N, 
                                g_gpu_telemetry.t4 / N, g_gpu_telemetry.t5 / N, g_gpu_telemetry.t6 / N, g_gpu_telemetry.t7 / N, 
                                g_gpu_telemetry.t8 / N, telemetry.t9 / N, telemetry.t10 / N, telemetry.t_total / N, g_numParticles);
                    std::fflush(stdout);
                    w.running = false;
                } else if (scene.frameNr > 60) {
                    telemetry.t9 += renderMs; telemetry.t10 += t10Ms; telemetry.t_total += totalMs;
                }
            } else {
                telemetry.t9 += renderMs; telemetry.t10 += t10Ms; telemetry.t_total += totalMs;
                if (g_gpu_telemetry.frames >= 60) {
                    float N = 60.0f;
                    std::printf("[CUDA] frame=%ld res=%d T1=%.3fms T2=%.3fms T3=%.3fms T4=%.3fms T5=%.3fms T6=%.3fms T7=%.3fms T8=%.3fms T9=%.3fms T10=%.3fms T_total=%.3fms particles=%d\n",
                                scene.frameNr, scene.resolution, g_gpu_telemetry.t1 / N, g_gpu_telemetry.t2 / N, g_gpu_telemetry.t3 / N, 
                                g_gpu_telemetry.t4 / N, g_gpu_telemetry.t5 / N, g_gpu_telemetry.t6 / N, g_gpu_telemetry.t7 / N, 
                                g_gpu_telemetry.t8 / N, telemetry.t9 / N, telemetry.t10 / N, telemetry.t_total / N, g_numParticles);
                    std::fflush(stdout); 
                    g_gpu_telemetry.reset(); telemetry.reset();
                }
            }
            scene.frameNr++;
        }

        fpsFrames++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fpsT0).count();
        if (elapsed >= 0.5) { lastFps = fpsFrames / elapsed; fpsT0 = now; fpsFrames = 0; }
    }

    cleanUp();
    cudaEventDestroy(evFrameStart); cudaEventDestroy(evFrameStop);
    cudaEventDestroy(evMapStart); cudaEventDestroy(evMapStop); 
    cudaEventDestroy(evUnmapStart); cudaEventDestroy(evUnmapStop);
    cudaEventDestroy(evRenderStart); cudaEventDestroy(evRenderStop);
    return 0;
}
