/*
 * CIFAR-10 Data Loader - Header File
 */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000
#define IMAGE_SIZE 3072  // 32 * 32 * 3
#define NUM_CLASSES 10
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_CHANNELS 3

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

// Function declarations
CIFAR10Dataset* initCIFAR10Dataset(const char* data_dir, bool use_cuda);
void freeCIFAR10Dataset(CIFAR10Dataset* dataset);
void printDatasetInfo(CIFAR10Dataset* dataset);
void shuffleTrainingData(CIFAR10Dataset* dataset);
void getBatch(CIFAR10Dataset* dataset, int batch_size, float* batch_images, 
              int* batch_labels, bool shuffle);

// Getter functions
float* getTrainImages(CIFAR10Dataset* dataset);
int* getTrainLabels(CIFAR10Dataset* dataset);
float* getTestImages(CIFAR10Dataset* dataset);
int* getTestLabels(CIFAR10Dataset* dataset);
int getNumTrainSamples(CIFAR10Dataset* dataset);
int getNumTestSamples(CIFAR10Dataset* dataset);

#ifdef __cplusplus
}
#endif

#endif // DATA_LOADER_H
