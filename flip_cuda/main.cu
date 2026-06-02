// flip_cuda/main.cu
// Entry point utama untuk simulasi fluida FLIP CUDA dengan render interop OpenGL (Node I2 & J)

#define GL_GLEXT_PROTOTYPES
#include "flip_fluid.cuh"
#include "device_data.cuh"
#include "gl_render_pipeline.h"
#include "ui.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

// Definisikan parameter d_params yang akan disalin ke constant memory GPU
__constant__ SimParams d_params;

// Struktur telemetry untuk pencatatan performa CUDA (Node J)
struct GpuTelemetry {
    float t1 = 0.0f; // Total waktu simulasi (fisika)
    float t_total = 0.0f; // Total waktu satu frame lengkap
    int frames = 0;

    void reset() {
        t1 = t_total = 0.0f;
        frames = 0;
    }
} g_gpuTelemetry;

// Konfigurasi Scene Global
struct Scene {
    float gravity          = -9.81f;
    float dt               = 1.0f / 60.0f;
    float flipRatio        = 0.9f;
    int   numPressureIters = 50;
    int   numParticleIters = 2;
    long  frameNr          = 0;
    float overRelaxation   = 1.9f;
    bool  compensateDrift  = true;
    bool  separateParticles = true;
    float obstacleX        = 3.0f; // Posisi default benchmark statis
    float obstacleY        = 2.0f;
    float obstacleRadius   = 0.15f;
    bool  paused           = false;
    bool  showGrid         = true;
    int   resolution       = 100; // Resolusi default grid
} scene;

// Struktur pembantu windowing X11 & GLX Context
struct WindowX11 {
    Display* dpy = nullptr;
    Window xwin  = 0;
    GLXContext glc = nullptr;
    int width = 1000;
    int height = 800;
} w;

// Deklarasi fungsi simulasi utama dari berkas cuda_fluid_simulator.cu (Node H)
extern void gpuSimulate(DeviceData& d, int numParticles, float dt, float gravity, 
                        float flipRatio, int numPressureIters, int numParticleIters, 
                        float overRelaxation, bool compensateDrift, bool separateParticles, 
                        float obstacleX, float obstacleY, float obstacleRadius, 
                        float obstacleVelX, float obstacleVelY, int numSubSteps);

// Deklarasi fungsi pendukung interop dari cuda_gl_interop.cu (Node I2)
extern void interopInit(DeviceData& d, const RenderPipeline& rp);
extern void interopMapResources(DeviceData& d);
extern void interopUnmapResources(DeviceData& d);
extern void interopDestroy(DeviceData& d);

// Global data simulasi
DeviceData d;
RenderPipeline rp;

// Vektor penyimpanan data di Host untuk penyemaian (seeding) awal
std::vector<float> h_posX;
std::vector<float> h_posY;
std::vector<float> h_s;
int numParticles = 0;

// Grid parameter salinan host
int g_fNumX = 0;
int g_fNumY = 0;
float g_h = 0.0f;
int g_fNumCells = 0;

static int s_glxAttrs[] = {
    GLX_RGBA,
    GLX_DOUBLEBUFFER,
    GLX_DEPTH_SIZE, 24,
    None
};

// Pembuat window X11 dan penginisiasi OpenGL visual
void createWindow() {
    w.dpy = XOpenDisplay(nullptr);
    if (!w.dpy) {
        std::cerr << "[X11] Gagal membuka koneksi X Display!" << std::endl;
        std::exit(1);
    }

    XVisualInfo* vi = glXChooseVisual(w.dpy, DefaultScreen(w.dpy), s_glxAttrs);
    if (!vi) {
        std::cerr << "[GLX] Tidak ada visual double buffer RGBA yang cocok!" << std::endl;
        std::exit(1);
    }

    Colormap cmap = XCreateColormap(w.dpy, RootWindow(w.dpy, vi->screen), vi->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;

    w.xwin = XCreateWindow(w.dpy, RootWindow(w.dpy, vi->screen), 0, 0, w.width, w.height, 0,
                           vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);

    XStoreName(w.dpy, w.xwin, "FLIP Fluid Simulator (GPU CUDA Engine - Brian)");
    XMapWindow(w.dpy, w.xwin);

    w.glc = glXCreateContext(w.dpy, vi, nullptr, GL_TRUE);
    if (!w.glc) {
        std::cerr << "[GLX] Gagal membuat OpenGL context!" << std::endl;
        std::exit(1);
    }
    glXMakeCurrent(w.dpy, w.xwin, w.glc);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Matikan VSync secara paksa jika didukung oleh ekstensi driver grafis MESA
    typedef void (*PFNGLXSWAPINTERVALMESAPROC)(unsigned int interval);
    PFNGLXSWAPINTERVALMESAPROC glXSwapIntervalMESA = 
        (PFNGLXSWAPINTERVALMESAPROC)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalMESA");
    if (glXSwapIntervalMESA) {
        glXSwapIntervalMESA(0); // Nonaktifkan sinkronisasi vertikal monitor
        std::cout << "[X11] VSync berhasil dimatikan secara paksa via MESA Swap Interval." << std::endl;
    }
}

// Inisialisasi seeding partikel dan pembuatan parameter grid
void setupScene() {
    float width = 6.0f;
    float height = 4.0f;
    float spacing = 4.0f / (float)scene.resolution;
    float particleRadius = 0.3f * spacing;

    g_fNumX = int(std::floor(width / spacing)) + 1;
    g_fNumY = int(std::floor(height / spacing)) + 1;
    g_h = std::max(width / (float)g_fNumX, height / (float)g_fNumY);
    float fInvSpacing = 1.0f / g_h;
    g_fNumCells = g_fNumX * g_fNumY;

    // Parameter constant spasial hash
    float pSpacing = 2.0f * particleRadius;
    float pInvSpacing = 1.0f / pSpacing;
    int pNumX = int(std::floor(width / pSpacing)) + 1;
    int pNumY = int(std::floor(height / pSpacing)) + 1;
    int pNumCells = pNumX * pNumY;

    int maxParticles = 4 * g_fNumCells;

    // Buat konfigurasi host_params untuk di-upload ke constant memory GPU
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
    hostParams.particleRadius = particleRadius;
    hostParams.density = 1000.0f;
    hostParams.maxParticles = maxParticles;

    // Upload parameter fisika ke Constant Memory GPU d_params
    cudaMemcpyToSymbol(d_params, &hostParams, sizeof(SimParams));

    // Seeding partikel di dalam ruang simulasi
    h_posX.clear();
    h_posY.clear();
    h_s.assign(g_fNumCells, 1.0f); // 1.0f = Fluid/Air, 0.0f = Solid rintangan/dinding

    float dx = 2.0f * particleRadius;
    float dy = 2.0f * particleRadius;

    // Batas dinding padat (Solid walls) di sekeliling grid simulasi
    for (int i = 0; i < g_fNumX; ++i) {
        for (int j = 0; j < g_fNumY; ++j) {
            if (i == 0 || i == g_fNumX - 1 || j == 0 || j == g_fNumY - 1) {
                h_s[i * g_fNumY + j] = 0.0f; // Set solid
            }
        }
    }

    // Seeding partikel membentuk balok air di tengah grid
    for (float px = 2.0f * g_h; px < 3.0f; px += dx) {
        for (float py = 2.0f * g_h; py < 3.0f; py += dy) {
            h_posX.push_back(px);
            h_posY.push_back(py);
        }
    }
    numParticles = h_posX.size();

    // Alokasikan memori GPU
    d.allocate(g_fNumCells, pNumCells, maxParticles);

    // Inisialisasi pipeline render OpenGL (Node I1)
    renderInit(rp, maxParticles, g_fNumCells, g_fNumX, g_fNumY, g_h);

    // Daftarkan buffer interop (Node I2)
    interopInit(d, rp);

    // Reset posisi awal partikel langsung ke VBO grafis yang terpetakan
    d.reset(h_posX.data(), h_posY.data(), h_s.data(), numParticles, g_fNumCells);
}

void cleanUp() {
    interopDestroy(d);
    renderDestroy(rp);
    d.free();
    if (w.glc) {
        glXMakeCurrent(w.dpy, None, nullptr);
        glXDestroyContext(w.dpy, w.glc);
    }
    if (w.xwin) XDestroyWindow(w.dpy, w.xwin);
    if (w.dpy)  XCloseDisplay(w.dpy);
}

int main(int argc, char** argv) {
    bool vsyncOff = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-vsync") == 0) {
            vsyncOff = true;
        }
    }

    std::cout << "[CUDA FLIP] Memulai inisialisasi simulator..." << std::endl;
    createWindow();
    setupScene();

    // Event markers untuk benchmark presisi (Node J)
    cudaEvent_t startTotal, stopTotal;
    cudaEvent_t startSim, stopSim;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startSim);
    cudaEventCreate(&stopSim);

    bool quit = false;
    XEvent xev;
    flipcpu_ui::Input uiIn;

    // Mouse drag states untuk memanipulasi rintangan lingkaran
    bool dragObstacle = false;
    float prevObstacleX = scene.obstacleX;
    float prevObstacleY = scene.obstacleY;

    std::cout << "[CUDA FLIP] Memulai game loop..." << std::endl;

    while (!quit) {
        cudaEventRecord(startTotal, 0);

        // 1. Tangani Event X11 (Keyboard & Mouse)
        while (XPending(w.dpy)) {
            XNextEvent(w.dpy, &xev);
            if (xev.type == KeyPress) {
                KeySym key = XLookupKeysym(&xev.xkey, 0);
                if (key == XK_Escape || key == XK_q) {
                    quit = true;
                } else if (key == XK_space || key == XK_p) {
                    scene.paused = !scene.paused;
                } else if (key == XK_g) {
                    scene.showGrid = !scene.showGrid;
                } else if (key == XK_r) {
                    // Reset Scene
                    interopDestroy(d);
                    renderDestroy(rp);
                    d.free();
                    setupScene();
                }
            } else if (xev.type == ButtonPress) {
                if (xev.xbutton.button == 1) { // Klik kiri mouse
                    float mx = (float)xev.xbutton.x / (float)w.width * 6.0f;
                    float my = (1.0f - (float)xev.xbutton.y / (float)w.height) * 4.0f;
                    float dx = mx - scene.obstacleX;
                    float dy = my - scene.obstacleY;
                    if (dx*dx + dy*dy < scene.obstacleRadius * scene.obstacleRadius) {
                        dragObstacle = true;
                        prevObstacleX = scene.obstacleX;
                        prevObstacleY = scene.obstacleY;
                    }
                }
            } else if (xev.type == ButtonRelease) {
                if (xev.xbutton.button == 1) {
                    dragObstacle = false;
                }
            } else if (xev.type == MotionNotify) {
                if (dragObstacle) {
                    scene.obstacleX = (float)xev.xmotion.x / (float)w.width * 6.0f;
                    scene.obstacleY = (1.0f - (float)xev.xmotion.y / (float)w.height) * 4.0f;
                }
            }
        }

        // Hitung kecepatan obstacle saat ditarik mouse
        float obstacleVelX = 0.0f;
        float obstacleVelY = 0.0f;
        if (dragObstacle && !scene.paused) {
            obstacleVelX = (scene.obstacleX - prevObstacleX) / scene.dt;
            obstacleVelY = (scene.obstacleY - prevObstacleY) / scene.dt;
            prevObstacleX = scene.obstacleX;
            prevObstacleY = scene.obstacleY;
        }

        // 2. Fase Simulasi Fisika CUDA (T1-T7) - Zero Host Transit
        float simMs = 0.0f;
        if (!scene.paused) {
            cudaEventRecord(startSim, 0);

            // MAP RESOURCE: Kaitkan pointer d.posX dkk langsung ke buffer OpenGL VBO
            interopMapResources(d);

            // Jalankan siklus perhitungan fluida penuh di GPU
            gpuSimulate(d, numParticles, scene.dt, scene.gravity, scene.flipRatio,
                        scene.numPressureIters, scene.numParticleIters, scene.overRelaxation,
                        scene.compensateDrift, scene.separateParticles, scene.obstacleX,
                        scene.obstacleY, scene.obstacleRadius, obstacleVelX, obstacleVelY, 1);

            // UNMAP RESOURCE: Kembalikan kepemilikan buffer ke OpenGL untuk rendering
            interopUnmapResources(d);

            cudaEventRecord(stopSim, 0);
            cudaEventSynchronize(stopSim);
            cudaEventElapsedTime(&simMs, startSim, stopSim);
        }

        // 3. Fase Rendering OpenGL (T9 - Render Stage)
        glClear(GL_COLOR_BUFFER_BIT);

        // Merender visualisasi MAC Grid (T9)
        if (scene.showGrid) {
            std::vector<float> h_cellColor(g_fNumCells * 3);
            cudaMemcpy(h_cellColor.data(), d.cellColor, g_fNumCells * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            renderGrid(rp, h_cellColor.data(), g_fNumX, g_fNumY, g_h);
        }

        // Merender partikel fluida langsung dari VBO interop (0ms transfer!)
        renderParticles(rp, numParticles, 6.0f);

        // Merender rintangan lingkaran
        renderObstacle(rp, scene.obstacleX, scene.obstacleY, scene.obstacleRadius);

        // 4. Update dan Render UI Overlay
        glUseProgram(0); // Matikan shader aktif sebelum menggambar immediate mode UI
        uiIn.screenW = w.width;
        uiIn.screenH = w.height;
        flipcpu_ui::begin(uiIn);
        flipcpu_ui::beginPanel(10, 10, 260, 280, "CUDA FLIP Simulator");
        flipcpu_ui::text("Particles: %d", numParticles);
        flipcpu_ui::text("Sim Time: %.2f ms", simMs);
        flipcpu_ui::checkbox("Separate Particles", &scene.separateParticles);
        flipcpu_ui::checkbox("Compensate Drift", &scene.compensateDrift);
        flipcpu_ui::slider("FLIP Ratio", &scene.flipRatio, 0.0f, 1.0f);
        if (flipcpu_ui::button("Reset Scene")) {
            interopDestroy(d);
            renderDestroy(rp);
            d.free();
            setupScene();
        }
        flipcpu_ui::endPanel();

        glXSwapBuffers(w.dpy, w.xwin);

        // 5. Akumulasi dan Cetak Telemetry Benchmark (Node J)
        cudaEventRecord(stopTotal, 0);
        cudaEventSynchronize(stopTotal);
        float totalMs = 0.0f;
        cudaEventElapsedTime(&totalMs, startTotal, stopTotal);

        if (!scene.paused) {
            g_gpuTelemetry.t_total += totalMs;
            g_gpuTelemetry.t1 += simMs;
            g_gpuTelemetry.frames++;

            if (g_gpuTelemetry.frames >= 60) {
                float N = 60.0f;
                std::printf("[CUDA] frame=%ld res=%d T_sim=%.3fms T_render=%.3fms T_total=%.3fms particles=%d\n",
                            scene.frameNr, scene.resolution,
                            g_gpuTelemetry.t1 / N,
                            (g_gpuTelemetry.t_total - g_gpuTelemetry.t1) / N,
                            g_gpuTelemetry.t_total / N, numParticles);
                std::fflush(stdout);
                g_gpuTelemetry.reset();
            }
        }
        scene.frameNr++;
    }

    cleanUp();
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    cudaEventDestroy(startSim);
    cudaEventDestroy(stopSim);

    return 0;
}