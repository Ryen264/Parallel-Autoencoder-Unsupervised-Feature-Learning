# CIFAR-10 Autoencoder + SVM Classification Pipeline
# Makefile for CUDA compilation

# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++14 -arch=sm_50 -O3
INCLUDE_DIRS = -I./include
LIBSVM_DIR = ./libsvm
LIBSVM_LIB = -L$(LIBSVM_DIR) -lsvm

# Source files
SRC_DIR = src
CPU_SRC_DIR = $(SRC_DIR)/cpu
SOURCES = $(SRC_DIR)/main.cu \
          $(SRC_DIR)/data_loader.cu \
          $(SRC_DIR)/autoencoder.cu \
          $(SRC_DIR)/dataset.cu
CPU_SOURCES = $(wildcard $(CPU_SRC_DIR)/*.cu)

# Object files
OBJECTS = $(SOURCES:.cu=.o)
CPU_OBJECTS = $(CPU_SOURCES:.cu=.o)

# Output executable
TARGET = pipeline

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS) $(CPU_OBJECTS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(CPU_OBJECTS) -o $(TARGET) $(LIBSVM_LIB)
	@echo "Build complete: $(TARGET)"

# Compile .cu files to .o
%.o: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Build without LIBSVM (for testing without SVM)
no-svm: LIBSVM_LIB =
no-svm: $(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(CPU_OBJECTS) $(TARGET) $(TARGET).exe
	@echo "Clean complete"

# Run the pipeline
run: $(TARGET)
	@echo "Running pipeline..."
	./$(TARGET)

# Help target
help:
	@echo "CIFAR-10 Autoencoder Pipeline - Makefile Commands"
	@echo "=================================================="
	@echo "make          - Build the complete pipeline"
	@echo "make no-svm   - Build without LIBSVM (placeholder SVM)"
	@echo "make clean    - Remove all build artifacts"
	@echo "make run      - Build and run the pipeline"
	@echo "make help     - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - CUDA Toolkit installed"
	@echo "  - LIBSVM library (optional, for SVM training)"
	@echo ""
	@echo "Installation:"
	@echo "  1. Install CUDA: https://developer.nvidia.com/cuda-downloads"
	@echo "  2. Install LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/"
	@echo "     - Extract to ./libsvm/"
	@echo "     - Run 'make' in libsvm directory"

# Phony targets
.PHONY: all clean run help no-svm

# Windows-specific adjustments (uncomment if building on Windows with MinGW)
# TARGET = pipeline.exe
# RM = del /Q
# LIBSVM_LIB = -L$(LIBSVM_DIR) -llibsvm
