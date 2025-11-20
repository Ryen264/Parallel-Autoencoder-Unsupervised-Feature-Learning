/*
 * CIFAR-10 Data Loader - CUDA Implementation
 * 
 * Data Loading and Preprocessing:
 * + Create a CIFAR10 Dataset structure to handle data loading 
 * + Read CIFAR-10 binary files (5 training batches + 1 test batch)
 * + Parse the binary format: 1 byte label + 3,072 bytes image per record 
 * + Convert uint8 pixel values [0, 255] to float [0, 1] for normalization
 * + Implement batch generation for training
 * + Add data shuffling capability 
 * + Organize train images (50,000), test images (10,000), and their labels in memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000
#define IMAGE_SIZE 3072  // 32 * 32 * 3
#define NUM_CLASSES 10
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_CHANNELS 3

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CIFAR-10 Dataset structure
typedef struct {
    float* train_images;      // Host memory: 50000 x 3072
    int* train_labels;        // Host memory: 50000
    float* test_images;       // Host memory: 10000 x 3072
    int* test_labels;         // Host memory: 10000
    
    float* d_train_images;    // Device memory
    int* d_train_labels;      // Device memory
    float* d_test_images;     // Device memory
    int* d_test_labels;       // Device memory
    
    int* train_indices;       // For shuffling
    int current_index;
} CIFAR10Dataset;

// CUDA kernel for normalization: convert uint8 [0, 255] to float [0, 1]
__global__ void normalizeKernel(unsigned char* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / 255.0f;
    }
}

// Read a single CIFAR-10 binary file
void readBinaryFile(const char* filepath, unsigned char** raw_data, int num_samples) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    
    int record_size = 1 + IMAGE_SIZE;  // 1 byte label + 3072 bytes image
    int total_size = num_samples * record_size;
    
    *raw_data = (unsigned char*)malloc(total_size);
    if (!*raw_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    size_t read_size = fread(*raw_data, 1, total_size, file);
    if (read_size != total_size) {
        fprintf(stderr, "Error: Could not read complete file %s\n", filepath);
        free(*raw_data);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    fclose(file);
}

// Parse raw data and normalize using CUDA
void parseAndNormalize(unsigned char* raw_data, float* images, int* labels, 
                       int num_samples, bool use_cuda) {
    int record_size = 1 + IMAGE_SIZE;
    
    // Extract labels
    for (int i = 0; i < num_samples; i++) {
        labels[i] = (int)raw_data[i * record_size];
    }
    
    if (use_cuda) {
        // Allocate device memory for raw image data
        unsigned char* d_raw_images;
        float* d_images;
        int image_data_size = num_samples * IMAGE_SIZE;
        
        CUDA_CHECK(cudaMalloc(&d_raw_images, image_data_size * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_images, image_data_size * sizeof(float)));
        
        // Copy raw image data to device (skip labels)
        unsigned char* raw_images = (unsigned char*)malloc(image_data_size);
        for (int i = 0; i < num_samples; i++) {
            memcpy(raw_images + i * IMAGE_SIZE, 
                   raw_data + i * record_size + 1, 
                   IMAGE_SIZE);
        }
        
        CUDA_CHECK(cudaMemcpy(d_raw_images, raw_images, 
                             image_data_size * sizeof(unsigned char), 
                             cudaMemcpyHostToDevice));
        
        // Launch normalization kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (image_data_size + threadsPerBlock - 1) / threadsPerBlock;
        normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_raw_images, d_images, image_data_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy normalized data back to host
        CUDA_CHECK(cudaMemcpy(images, d_images, 
                             image_data_size * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Cleanup
        free(raw_images);
        CUDA_CHECK(cudaFree(d_raw_images));
        CUDA_CHECK(cudaFree(d_images));
    } else {
        // CPU normalization
        for (int i = 0; i < num_samples; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                images[i * IMAGE_SIZE + j] = 
                    raw_data[i * record_size + 1 + j] / 255.0f;
            }
        }
    }
}

// Initialize CIFAR-10 dataset
CIFAR10Dataset* initCIFAR10Dataset(const char* data_dir, bool use_cuda) {
    CIFAR10Dataset* dataset = (CIFAR10Dataset*)malloc(sizeof(CIFAR10Dataset));
    if (!dataset) {
        fprintf(stderr, "Error: Memory allocation failed for dataset\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate host memory
    dataset->train_images = (float*)malloc(NUM_TRAIN_SAMPLES * IMAGE_SIZE * sizeof(float));
    dataset->train_labels = (int*)malloc(NUM_TRAIN_SAMPLES * sizeof(int));
    dataset->test_images = (float*)malloc(NUM_TEST_SAMPLES * IMAGE_SIZE * sizeof(float));
    dataset->test_labels = (int*)malloc(NUM_TEST_SAMPLES * sizeof(int));
    dataset->train_indices = (int*)malloc(NUM_TRAIN_SAMPLES * sizeof(int));
    
    if (!dataset->train_images || !dataset->train_labels || 
        !dataset->test_images || !dataset->test_labels || !dataset->train_indices) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize train indices
    for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
        dataset->train_indices[i] = i;
    }
    dataset->current_index = 0;
    
    // Load training data (5 batches)
    printf("Loading training data...\n");
    for (int batch = 1; batch <= 5; batch++) {
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "%s/data_batch_%d.bin", data_dir, batch);
        
        unsigned char* raw_data;
        readBinaryFile(filepath, &raw_data, 10000);
        
        int offset = (batch - 1) * 10000;
        parseAndNormalize(raw_data, 
                         dataset->train_images + offset * IMAGE_SIZE,
                         dataset->train_labels + offset,
                         10000, use_cuda);
        
        free(raw_data);
        printf("  Loaded batch %d/5\n", batch);
    }
    
    // Load test data
    printf("Loading test data...\n");
    char test_filepath[256];
    snprintf(test_filepath, sizeof(test_filepath), "%s/test_batch.bin", data_dir);
    
    unsigned char* raw_data;
    readBinaryFile(test_filepath, &raw_data, NUM_TEST_SAMPLES);
    parseAndNormalize(raw_data, dataset->test_images, dataset->test_labels, 
                     NUM_TEST_SAMPLES, use_cuda);
    free(raw_data);
    printf("  Test data loaded\n");
    
    // Allocate device memory if using CUDA
    if (use_cuda) {
        CUDA_CHECK(cudaMalloc(&dataset->d_train_images, 
                             NUM_TRAIN_SAMPLES * IMAGE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dataset->d_train_labels, 
                             NUM_TRAIN_SAMPLES * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dataset->d_test_images, 
                             NUM_TEST_SAMPLES * IMAGE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dataset->d_test_labels, 
                             NUM_TEST_SAMPLES * sizeof(int)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(dataset->d_train_images, dataset->train_images,
                             NUM_TRAIN_SAMPLES * IMAGE_SIZE * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dataset->d_train_labels, dataset->train_labels,
                             NUM_TRAIN_SAMPLES * sizeof(int),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dataset->d_test_images, dataset->test_images,
                             NUM_TEST_SAMPLES * IMAGE_SIZE * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dataset->d_test_labels, dataset->test_labels,
                             NUM_TEST_SAMPLES * sizeof(int),
                             cudaMemcpyHostToDevice));
    } else {
        dataset->d_train_images = NULL;
        dataset->d_train_labels = NULL;
        dataset->d_test_images = NULL;
        dataset->d_test_labels = NULL;
    }
    
    return dataset;
}

// Shuffle training data indices
void shuffleTrainingData(CIFAR10Dataset* dataset) {
    srand(time(NULL));
    for (int i = NUM_TRAIN_SAMPLES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = dataset->train_indices[i];
        dataset->train_indices[i] = dataset->train_indices[j];
        dataset->train_indices[j] = temp;
    }
    dataset->current_index = 0;
}

// Get a batch of training data
void getBatch(CIFAR10Dataset* dataset, int batch_size, float* batch_images, 
              int* batch_labels, bool shuffle) {
    // Check if we need to start a new epoch
    if (dataset->current_index + batch_size > NUM_TRAIN_SAMPLES) {
        if (shuffle) {
            shuffleTrainingData(dataset);
        } else {
            dataset->current_index = 0;
        }
    }
    
    // Copy batch data
    for (int i = 0; i < batch_size; i++) {
        int idx = dataset->train_indices[dataset->current_index + i];
        memcpy(batch_images + i * IMAGE_SIZE,
               dataset->train_images + idx * IMAGE_SIZE,
               IMAGE_SIZE * sizeof(float));
        batch_labels[i] = dataset->train_labels[idx];
    }
    
    dataset->current_index += batch_size;
}

// Free dataset memory
void freeCIFAR10Dataset(CIFAR10Dataset* dataset) {
    if (dataset) {
        free(dataset->train_images);
        free(dataset->train_labels);
        free(dataset->test_images);
        free(dataset->test_labels);
        free(dataset->train_indices);
        
        if (dataset->d_train_images) {
            cudaFree(dataset->d_train_images);
            cudaFree(dataset->d_train_labels);
            cudaFree(dataset->d_test_images);
            cudaFree(dataset->d_test_labels);
        }
        
        free(dataset);
    }
}

// Print dataset information
void printDatasetInfo(CIFAR10Dataset* dataset) {
    printf("\n");
    printf("============================================================\n");
    printf("CIFAR-10 Dataset Information\n");
    printf("============================================================\n");
    printf("  Number of training samples: %d\n", NUM_TRAIN_SAMPLES);
    printf("  Number of test samples: %d\n", NUM_TEST_SAMPLES);
    printf("  Image dimensions: %dx%dx%d\n", IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS);
    printf("  Number of classes: %d\n", NUM_CLASSES);
    printf("  Total image size: %d bytes\n", IMAGE_SIZE);
    
    // Calculate min/max for verification
    float train_min = dataset->train_images[0];
    float train_max = dataset->train_images[0];
    float test_min = dataset->test_images[0];
    float test_max = dataset->test_images[0];
    
    for (int i = 0; i < NUM_TRAIN_SAMPLES * IMAGE_SIZE; i++) {
        if (dataset->train_images[i] < train_min) train_min = dataset->train_images[i];
        if (dataset->train_images[i] > train_max) train_max = dataset->train_images[i];
    }
    
    for (int i = 0; i < NUM_TEST_SAMPLES * IMAGE_SIZE; i++) {
        if (dataset->test_images[i] < test_min) test_min = dataset->test_images[i];
        if (dataset->test_images[i] > test_max) test_max = dataset->test_images[i];
    }
    
    printf("\n");
    printf("============================================================\n");
    printf("Verification Results\n");
    printf("============================================================\n");
    printf("✓ Training images: %d samples\n", NUM_TRAIN_SAMPLES);
    printf("✓ Test images: %d samples\n", NUM_TEST_SAMPLES);
    printf("✓ Preprocessing - Normalized to [0, 1]:\n");
    printf("  - Training data range: [%.2f, %.2f]\n", train_min, train_max);
    printf("  - Test data range: [%.2f, %.2f]\n", test_min, test_max);
}

// Main function
int main(int argc, char** argv) {
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    bool use_cuda = (error == cudaSuccess && deviceCount > 0);
    
    if (use_cuda) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\nCUDA Device Found: %s\n", prop.name);
        printf("Using CUDA for data normalization\n");
    } else {
        printf("\nNo CUDA device found. Using CPU mode.\n");
    }
    
    // Data directory
    const char* data_dir = "/content/data/cifar-10-batches-bin";
    
    // Load dataset
    CIFAR10Dataset* dataset = initCIFAR10Dataset(data_dir, use_cuda);
    printf("\n✓ Dataset loaded successfully!\n");
    
    // Print dataset information and verification
    printDatasetInfo(dataset);
    
    // Test batch generation
    printf("\n============================================================\n");
    printf("Batch Generation Test\n");
    printf("============================================================\n");
    
    int batch_size = 128;
    float* batch_images = (float*)malloc(batch_size * IMAGE_SIZE * sizeof(float));
    int* batch_labels = (int*)malloc(batch_size * sizeof(int));
    
    getBatch(dataset, batch_size, batch_images, batch_labels, true);
    
    printf("  Batch size: %d\n", batch_size);
    printf("  Batch images shape: (%d, %d)\n", batch_size, IMAGE_SIZE);
    printf("  Sample labels: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", batch_labels[i]);
    }
    printf("\n");
    
    // Calculate batch pixel range
    float batch_min = batch_images[0];
    float batch_max = batch_images[0];
    for (int i = 0; i < batch_size * IMAGE_SIZE; i++) {
        if (batch_images[i] < batch_min) batch_min = batch_images[i];
        if (batch_images[i] > batch_max) batch_max = batch_images[i];
    }
    printf("  Batch pixel range: [%.2f, %.2f]\n", batch_min, batch_max);
    
    // Cleanup
    free(batch_images);
    free(batch_labels);
    freeCIFAR10Dataset(dataset);
    
    return 0;
}
