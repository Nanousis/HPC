# Unified Makefile: CPU (GCC) + GPU (CUDA/NVCC)

########################
# CPU (GCC) configuration
########################
CPU_CC      := gcc
CPU_CFLAGS  := -Wall -Wextra -O3 -D_GNU_SOURCE
CPU_LDFLAGS := 
CPU_LDLIBS  := -fopenmp -lm

CPU_SRC     := nbody.c
CPU_OBJ     := $(CPU_SRC:%.c=build/cpu/%.o)
CPU_TARGET  := nbody_cpu

########################
# GPU (CUDA) configuration
########################
GPU_NVCC     := nvcc
GPU_CFLAGS   := -O4  -D_GNU_SOURCE -lineinfo
GPU_LDFLAGS  := 
GPU_LDLIBS   := -Xcompiler -fopenmp -lm

GPU_SRC      := nbody.cu
GPU_OBJ      := $(GPU_SRC:%.cu=build/gpu/%.o)
GPU_TARGET   := nbody_gpu


GALAXY_DATA_DIR   := ./galaxy_data.bin

########################
# Phony targets
########################
.PHONY: all cpu gpu clean run_cpu run_gpu help

all: $(CPU_TARGET) $(GPU_TARGET)
cpu: $(CPU_TARGET)
gpu: $(GPU_TARGET)
help:
	@echo "make all       # build CPU and GPU"
	@echo "make cpu       # build CPU only"
	@echo "make gpu       # build GPU only"
	@echo "make run_cpu   # run clahe_cpu"
	@echo "make run_gpu   # run clahe_gpu"
	@echo "make clean     # remove builds and binaries"

########################
# Linking
########################
$(CPU_TARGET): $(CPU_OBJ)
	$(CPU_CC) $(CPU_LDFLAGS) $(CPU_OBJ) -o $@ $(CPU_LDLIBS)

$(GPU_TARGET): $(GPU_OBJ)
	$(GPU_NVCC) $(GPU_LDFLAGS) $(GPU_OBJ) -o $@ $(GPU_LDLIBS)

########################
# Compilation rules
########################
# Ensure build directories exist
build/cpu build/gpu:
	@mkdir -p $@

# CPU objects
build/cpu/%.o: %.c | build/cpu
	$(CPU_CC) $(CPU_CFLAGS) -c $< -o $@ $(CPU_LDLIBS)

# GPU objects
build/gpu/%.o: %.cu | build/gpu
	$(GPU_NVCC) $(GPU_CFLAGS) -c $< -o $@ $(GPU_LDLIBS)

########################
# Run helpers
########################
run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) $(GALAXY_DATA_DIR)

run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) $(GALAXY_DATA_DIR)

########################
# Cleanup
########################
clean:
	rm -rf build $(CPU_TARGET) $(GPU_TARGET)
