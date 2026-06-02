# Usage:
#   make cpu        build flip_cpu/flip
#   make cuda       build flip_cuda/flip
#   make run-cpu
#   make run-cuda
#   make clean-cpu / clean-cuda / clean
#   GPU arch: change SM for your card (sm_86=RTX30xx, sm_89=RTX40xx, sm_75=GTX16xx)

SM ?= native

CXX       := g++
CXXFLAGS  := -std=c++17 -O3 -ffast-math -Wall -Wextra -mavx2 -mfma -fopenmp
CPU_LIBS  := -lGL -lGLX -lX11 -lm -lgomp

NVCC      := nvcc
NVCCFLAGS := -std=c++17 -O3 -arch=$(SM) --use_fast_math -rdc=true
CUDA_HOME ?= /usr/local/cuda
CUDA_LIBS := -lGL -lGLX -lX11 -lm -lcudart

CPU_DIR := flip_cpu
CUD_DIR := flip_cuda

CPU_SRC := $(CPU_DIR)/flip_fluid.cpp $(CPU_DIR)/ui.cpp $(CPU_DIR)/main.cpp
CPU_OBJ := $(CPU_SRC:.cpp=.o)
CPU_BIN := $(CPU_DIR)/flip

CPU_OPT_OBJ := $(CPU_DIR)/flip_fluid_opt.o $(CPU_DIR)/ui_opt.o $(CPU_DIR)/main_opt.o
CPU_OPT_BIN := $(CPU_DIR)/flip_opt

CU_SRC  := $(CUD_DIR)/particle_kinematics.cu \
            $(CUD_DIR)/spatial_hash.cu \
            $(CUD_DIR)/p2g_transfer.cu \
            $(CUD_DIR)/g2p_transfer.cu \
            $(CUD_DIR)/pressure_solver.cu \
            $(CUD_DIR)/colors_reduce.cu \
            $(CUD_DIR)/cuda_gl_interop.cu \
            $(CUD_DIR)/device_data.cu \
            $(CUD_DIR)/cuda_fluid_simulator.cu \
            $(CUD_DIR)/main.cu
CPP_SRC := $(CUD_DIR)/gl_render_pipeline.cpp $(CUD_DIR)/ui.cpp
CU_OBJ := $(CU_SRC:.cu=.o)
CPP_OBJ := $(CPP_SRC:.cpp=.o)
CUD_BIN := $(CUD_DIR)/flip

.PHONY: cpu cpu-opt cuda run-cpu run-cpu-opt run-cuda clean-cpu clean-cuda clean

cpu: $(CPU_BIN)

$(CPU_BIN): $(CPU_OBJ)
	$(CXX) -o $@ $^ $(CPU_LIBS)

cpu-opt: $(CPU_OPT_BIN)

$(CPU_OPT_BIN): $(CPU_OPT_OBJ)
	$(CXX) -fopenmp -o $@ $^ $(CPU_LIBS)

$(CPU_DIR)/%.o: $(CPU_DIR)/%.cpp $(CPU_DIR)/flip_fluid.h $(CPU_DIR)/ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(CPU_DIR)/%_opt.o: $(CPU_DIR)/%.cpp $(CPU_DIR)/flip_fluid.h $(CPU_DIR)/ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(CPU_DIR)/flip_fluid_opt.o: $(CPU_DIR)/flip_fluid_opt.cpp $(CPU_DIR)/flip_fluid.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

cuda: $(CUD_BIN)

$(CUD_BIN): $(CU_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LIBS)

$(CUD_DIR)/%.o: $(CUD_DIR)/%.cu $(CUD_DIR)/flip_fluid.cuh $(CUD_DIR)/device_data.cuh
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

$(CUD_DIR)/%.o: $(CUD_DIR)/%.cpp
	$(CXX) -std=c++17 -O3 -ffast-math -I$(CUDA_HOME)/include -c -o $@ $<

run-cpu: cpu
	$(CPU_BIN)

run-cpu-opt: cpu-opt
	$(CPU_OPT_BIN)

run-cuda: cuda
	$(CUD_BIN)

clean-cpu:
	rm -f $(CPU_OBJ) $(CPU_OPT_OBJ) $(CPU_BIN) $(CPU_OPT_BIN)

clean-cuda:
	rm -f $(CU_OBJ) $(CPP_OBJ) $(CUD_BIN)

clean: clean-cpu clean-cuda
