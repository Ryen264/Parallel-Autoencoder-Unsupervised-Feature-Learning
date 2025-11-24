# CIFAR-10 Data Loader for Google Colab

This folder contains Google Colab-compatible versions of the CIFAR-10 data loader with adjusted file paths.

## Files

- **constants.h** - Constants for CIFAR-10 dataset
- **dataset.h** - Dataset structure and function declarations
- **data_loader.cu** - CUDA-accelerated data loading and preprocessing
- **test_data_loader.cu** - Comprehensive test program

## Usage in Google Colab

### 1. Upload Files to Colab

```python
# Upload these files to /content/ in Colab
from google.colab import files
uploaded = files.upload()  # Upload all .h and .cu files
```

### 2. Download CIFAR-10 Dataset

```python
# Download and extract CIFAR-10 binary dataset
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
!mv cifar-10-batches-bin /content/
```

### 3. Compile the Program

```bash
# Compile with nvcc
!nvcc -std=c++14 -arch=sm_75 -O2 \
  test_data_loader.cu data_loader.cu \
  -o test_data_loader
```

### 4. Run the Test

```bash
# Run with default path (/content/cifar-10-batches-bin)
!./test_data_loader

# Or specify custom path
!./test_data_loader /path/to/cifar-10-batches-bin
```

## Complete Colab Notebook Example

```python
# === Cell 1: Upload source files ===
from google.colab import files
print("Upload constants.h, dataset.h, data_loader.cu, test_data_loader.cu")
uploaded = files.upload()

# === Cell 2: Download CIFAR-10 dataset ===
!wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
!ls -lh cifar-10-batches-bin/

# === Cell 3: Compile ===
!nvcc -std=c++14 -arch=sm_75 -O2 \
  test_data_loader.cu data_loader.cu \
  -o test_data_loader

# === Cell 4: Run tests ===
!./test_data_loader
```

## Key Differences from Local Version

1. **Default data path**: Changed from `./data/cifar-10-batches-bin` to `/content/cifar-10-batches-bin`
2. **IS_TEST_DATA_LOADER**: Set to `true` by default for immediate testing
3. **All includes are local**: No relative path adjustments needed (all files in same directory)

## Modifying for Production Use

To use for actual training (disable verbose testing):

1. Edit `test_data_loader.cu`
2. Change line: `bool IS_TEST_DATA_LOADER = true;` to `false`
3. Recompile

Or modify programmatically:

```cpp
// In your main training code
extern bool IS_TEST_DATA_LOADER;
IS_TEST_DATA_LOADER = false;  // Disable test output
```

## Expected Output

When `IS_TEST_DATA_LOADER = true`:
- CUDA device information
- Loading progress (5 training batches + 1 test batch)
- 16 verification tests
- Statistical analysis (mean, std dev, class distribution)
- Final summary with pass/fail counts

When `IS_TEST_DATA_LOADER = false`:
- Simple loading messages
- Dataset loaded confirmation
- No verbose testing output
