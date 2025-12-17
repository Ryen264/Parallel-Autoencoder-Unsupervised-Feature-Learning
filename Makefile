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
GPU_TRAIN_TARGET = gpu_autoencoder_train

gpu_train: data_loader.o progress_bar.o timer.o utils.o $(GPU_OBJECTS)
	@mkdir -p $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET_DIR)/$(GPU_TRAIN_TARGET) $(OBJECTS) $(GPU_OBJECTS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(CONSTANTS) $(MACRO)
	@echo "Compiling $<..."
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

clean:
	@rm -rf $(OBJ_DIR) $(TARGET_DIR)
