# Unified Makefile: CPU (GCC) + GPU (CUDA/NVCC)

########################
# CPU (GCC) configuration
########################
CPU_CC      := gcc
CPU_CFLAGS  := -Wall -Wextra -O2 -std=c11
CPU_LDFLAGS :=
CPU_LDLIBS  := -lm

CPU_SRC     := main.c clahe.c
CPU_OBJ     := $(CPU_SRC:%.c=build/cpu/%.o)
CPU_TARGET  := clahe_cpu

########################
# GPU (CUDA) configuration
########################
GPU_NVCC     := nvcc
GPU_CFLAGS   := -O4 
GPU_LDFLAGS  :=
GPU_LDLIBS   := -lm

GPU_SRC      := main.cu clahe.cu
GPU_OBJ      := $(GPU_SRC:%.cu=build/gpu/%.o)
GPU_TARGET   := clahe_gpu


INPUT_IMAGE_DIR   := ../Images/senator.pgm
OUTPUT_IMAGE_NAME := output

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
build/cpu/%.o: %.c clahe.h | build/cpu
	$(CPU_CC) $(CPU_CFLAGS) -c $< -o $@

# GPU objects
build/gpu/%.o: %.cu clahe.h | build/gpu
	$(GPU_NVCC) $(GPU_CFLAGS) -c $< -o $@

# Special case: clahe.cu (add PTXAS options)
build/gpu/clahe.o: clahe.cu clahe.h | build/gpu
	$(GPU_NVCC) $(GPU_CFLAGS) --ptxas-options=-v -c $< -o $@
########################
# Run helpers
########################
run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) $(INPUT_IMAGE_DIR) $(OUTPUT_IMAGE_NAME)_cpu.pgm

run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) $(INPUT_IMAGE_DIR) $(OUTPUT_IMAGE_NAME)_gpu.pgm

########################
# Cleanup
########################
clean:
	rm -rf build $(CPU_TARGET) $(GPU_TARGET)
