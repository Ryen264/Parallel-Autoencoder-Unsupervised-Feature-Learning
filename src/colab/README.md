# Phase 1 Complete Pipeline for Google Colab

This folder contains Google Colab-compatible versions of the Phase 1 complete pipeline: data loading, autoencoder training, and feature extraction.

## Files

### Data Loading
- **constants.h** - Constants for CIFAR-10 dataset
- **data_loader.h** - Dataset structure and function declarations
- **data_loader.cu** - CUDA-accelerated data loading and preprocessing
- **test_data_loader.cu** - Comprehensive test program

### Autoencoder
- **autoencoder.h** - Base autoencoder interface class
- **autoencoder.cu** - Base autoencoder implementation
- **cpu_autoencoder.h** - CPU autoencoder class
- **cpu_autoencoder.cu** - CPU autoencoder implementation
- **cpu_layers.h** - CPU layer operations interface
- **cpu_layers.cu** - CPU layer operations (conv2D, pooling, etc.)

### Complete Pipeline Test
- **test_phase_1.cu** - Tests all 3 steps: load data, train autoencoder, extract features
- **test_phase_1.ipynb** - Jupyter notebook for running complete pipeline on Colab

## Usage in Google Colab

### 1. Upload Files to Colab

```python
# Upload these files to /content/ in Colab
from google.colab import files

# For data loader only test:
uploaded = files.upload()  # Upload: constants.h, data_loader.h, data_loader.cu, test_data_loader.cu

# For complete Phase 1 pipeline:
uploaded = files.upload()  # Upload all .h and .cu files (10 files total)
```

### 2. Download CIFAR-10 Dataset

```python
# Download and extract CIFAR-10 binary dataset
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
```

### 3A. Test Data Loader Only

```bash
# Compile with nvcc
!nvcc -arch=sm_75 \
  test_data_loader.cu data_loader.cu \
  -o test_data_loader


# Run the test
!./test_data_loader
```

### 3B. Test Complete Phase 1 Pipeline

```bash
# Compile all components
!nvcc -arch=sm_75 \
  test_phase_1.cu \
  data_loader.cu \
  autoencoder.cu \
  cpu_autoencoder.cu \
  cpu_layers.cu \
  -o test_phase_1

# Run with default parameters (3 epochs)
!./test_phase_1

# Or customize: ./test_phase_1 <data_dir> <epochs> <batch_size> <learning_rate>
!./test_phase_1 /content/cifar-10-batches-bin 5 128 0.001
```

## Quick Start with Jupyter Notebook

Use **test_phase_1.ipynb** for step-by-step execution:

1. Open `test_phase_1.ipynb` in Google Colab
2. Follow the instructions in each cell
3. Upload all required files (10 total)
4. Run cells sequentially to:
   - Check GPU capability
   - Download CIFAR-10
   - Compile the program
   - Run complete Phase 1 pipeline

## Key Differences from Local Version

1. **Default data path**: Changed from `./data/cifar-10-batches-bin` to `/content/cifar-10-batches-bin`
2. **Test flags**: Set to `true` by default for immediate testing
   - `IS_TEST_DATA_LOADER = true` in test_data_loader.cu
   - `IS_TEST_PHASE_1 = true` in test_phase_1.cu
3. **All includes are local**: No relative path adjustments needed (all files in same directory)

## Phase 1 Pipeline Overview

### Step 1: Load and Preprocess CIFAR-10
- Loads 50,000 training images
- Loads 10,000 test images
- Normalizes to [0, 1] range
- Verifies data integrity

### Step 2: Train Autoencoder (Unsupervised)
- Architecture: 32×32×3 → 8×8×128 (8192 features)
- Encoder: Conv2D(256) → MaxPool → Conv2D(128) → MaxPool
- Decoder: Conv2D(128) → Upsample → Conv2D(256) → Upsample → Conv2D(3)
- Training on 50K images (ignores labels)
- Monitors loss reduction

### Step 3: Extract Features
- Encodes training set: (50000, 8192)
- Encodes test set: (10000, 8192)
- Preserves labels for downstream tasks
- Ready for supervised learning (Phase 2)

## Expected Execution Time

On Google Colab (with GPU):
- **Step 1** (Data Loading): ~10-30 seconds
- **Step 2** (Training, 3 epochs): ~5-15 minutes
- **Step 3** (Feature Extraction): ~1-3 minutes
- **Total**: ~6-18 minutes for 3 epochs

Note: CPU-only execution will be significantly slower (30+ minutes).

## Troubleshooting

### Compilation Errors
- Ensure all 10 files are uploaded
- Check CUDA architecture matches your GPU (`-arch=sm_XX`)
- Use `!nvcc --version` to verify NVCC is available

### Memory Issues
- Reduce batch size (e.g., 64 instead of 128)
- Reduce number of epochs for testing
- Enable Colab GPU runtime (Runtime → Change runtime type → GPU)

### Data Loading Issues
- Verify dataset path: `!ls /content/cifar-10-batches-bin/`
- Check all 6 files exist: data_batch_1.bin through data_batch_5.bin, test_batch.bin
- Ensure batches.meta.txt exists
- Final summary with pass/fail counts

When `IS_TEST_DATA_LOADER = false`:
- Simple loading messages
- Dataset loaded confirmation
- No verbose testing output
