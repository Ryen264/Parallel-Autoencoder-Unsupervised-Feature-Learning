# CIFAR-10 Autoencoder + SVM Classification Pipeline
# Makefile for CUDA compilation

# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++20 -arch=sm_75 -O3 --expt-relaxed-constexpr -diag-suppress 3012
INCLUDE_DIRS = -I./include -I./include/cpu -I./include/gpu -I./cuml-new/cpp/include -I./raft/cpp/include -I./rmm/cpp/include

# Source files
SRC_DIR = src
GPU_SRC_DIR = src/gpu
SRC := $(shell find $(SRC_DIR) -maxdepth 1 -name '*.cu')
GPU_SRC := $(shell find $(GPU_SRC_DIR) -maxdepth 1 -name '*.cu')
CONSTANTS = include/constants.h
MACRO = include/macro.h

# Object files
OBJ_DIR = obj
OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC))
GPU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(GPU_SRC))

# Dependancies
DEPS := $(OBJECTS:.o=.d)
-include $(DEPS)

TARGET_DIR = bin
GPU_AUTOENCODER_DEPS = $(OBJ_DIR)/data_loader.o $(OBJ_DIR)/progress_bar.o $(OBJ_DIR)/timer.o $(OBJ_DIR)/utils.o $(GPU_OBJECTS)
GPU_AUTOENCODER_TARGET = gpu_autoencoder

gpu_autoencoder: $(GPU_AUTOENCODER_DEPS)
	@echo "Compiling gpu autoencoder..."
	@mkdir -p $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET_DIR)/$(GPU_AUTOENCODER_TARGET) $(GPU_AUTOENCODER_DEPS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(CONSTANTS) $(MACRO)
	@echo "Compiling $<..."
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

clean:
	@rm -rf $(OBJ_DIR) $(TARGET_DIR)
