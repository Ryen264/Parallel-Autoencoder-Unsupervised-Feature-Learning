# LIBSVM Setup Instructions

This project uses LIBSVM for SVM classification. Follow these steps to set it up:

## Installation

### Option 1: Clone from GitHub (Recommended)

```bash
git clone https://github.com/cjlin1/libsvm.git
```

This will create a `libsvm/` directory in your project root.

### Option 2: Download and Extract

1. Download LIBSVM from: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
2. Extract to your project directory as `libsvm/`

## Verify Installation

After installation, your project structure should include:

```
Project/
├── libsvm/
│   ├── svm.h
│   ├── svm.cpp
│   └── ... (other libsvm files)
├── include/
├── src/
└── Makefile
```

## Compilation

The Makefile is already configured to compile libsvm. Simply run:

```bash
make gpu_autoencoder
```

This will:
1. Compile libsvm/svm.cpp to libsvm/svm.o
2. Link it with your CUDA code
3. Create the executable in bin/gpu_autoencoder

## Usage

The SVMmodel class now uses LIBSVM for training and prediction:

```cpp
SVMmodel svm(10.0f, "RBF", "auto");  // C=10, RBF kernel, auto gamma
svm.train(train_data, train_labels);
vector<int> predictions = svm.predict(test_data);
```

## Key Features Implemented

- **train()**: Trains SVM using LIBSVM with configurable parameters
- **predict()**: Makes predictions on test data
- **save()**: Saves trained model to disk
- **load()**: Loads pre-trained model
- **calculateAccuracy()**: Computes classification accuracy
- **calculateConfusionMatrix()**: Generates confusion matrix

## Parameters

- **C**: Regularization parameter (default: 10.0)
- **kernel_type**: "RBF" or "LINEAR" (default: "RBF")
- **gamma_type**: "auto" or numeric value (default: "auto")
- **tolerance**: Stopping criterion (default: 1e-3)
- **cache_size**: Cache memory in MB (default: 200.0)

## Troubleshooting

If you encounter compilation errors:

1. Ensure libsvm directory exists in project root
2. Check that svm.h and svm.cpp are present
3. Verify your C++ compiler supports C++11 or later
