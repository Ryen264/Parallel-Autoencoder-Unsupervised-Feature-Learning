/*
 * Phase 1 Test Program (Google Colab Version)
 * 
 * Tests all three steps of Phase 1:
 * - Step 1: Load and preprocess CIFAR-10 dataset (50K train, 10K test)
 * - Step 2: Train autoencoder on 50,000 training images (unsupervised)
 * - Step 3: Extract features (50000, 8192) and (10000, 8192)
 * 
 * Default data path for Google Colab: /content/cifar-10-batches-bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include "data_loader.h"
#include "cpu_autoencoder.h"
#include "constants.h"

// Global flag to control test mode (enabled by default for Colab)
bool IS_TEST_PHASE_1 = true;

// Test statistics structure
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
} TestStats;

TestStats stats = {0, 0, 0};

// Print test result
void print_test(const char* name, bool passed) {
    if (!IS_TEST_PHASE_1) return;
    
    stats.total_tests++;
    if (passed) {
        stats.passed_tests++;
        printf("  [PASS] %s\n", name);
    } else {
        stats.failed_tests++;
        printf("  [FAIL] %s\n", name);
    }
}

// Test: Verify dataset sample count
bool test_sample_count(const Dataset& dataset, int expected, const char* name) {
    bool passed = (dataset.n == expected);
    char msg[256];
    snprintf(msg, sizeof(msg), "%s sample count: %d (expected %d)", 
             name, dataset.n, expected);
    print_test(msg, passed);
    return passed;
}

// Test: Verify normalization to [0,1]
bool test_normalization(const Dataset& dataset, const char* name) {
    float* data = dataset.get_data();
    int total_size = dataset.n * dataset.width * dataset.height * dataset.depth;
    
    float min_val = data[0];
    float max_val = data[0];
    
    for (int i = 0; i < total_size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    bool passed = (min_val >= 0.0f && min_val < 0.1f && max_val <= 1.0f && max_val > 0.9f);
    char msg[256];
    snprintf(msg, sizeof(msg), "%s normalized to [0,1]: range [%.4f, %.4f]", 
             name, min_val, max_val);
    print_test(msg, passed);
    return passed;
}

// Test: Verify encoded feature dimensions
bool test_encoded_dimensions(const Dataset& encoded, int expected_n, int expected_features, const char* name) {
    int actual_features = encoded.width * encoded.height * encoded.depth;
    bool passed = (encoded.n == expected_n && actual_features == expected_features);
    
    char msg[256];
    snprintf(msg, sizeof(msg), "%s dimensions: (%d, %d) - expected (%d, %d)", 
             name, encoded.n, actual_features, expected_n, expected_features);
    print_test(msg, passed);
    return passed;
}

// Test: Verify loss decreased during training
bool test_loss_decrease(float initial_loss, float final_loss) {
    bool passed = (final_loss < initial_loss * 0.95f);
    
    char msg[256];
    snprintf(msg, sizeof(msg), "Training loss decreased: %.4f -> %.4f (%.1f%% reduction)", 
             initial_loss, final_loss, 100.0f * (1 - final_loss/initial_loss));
    print_test(msg, passed);
    return passed;
}

// Check CUDA availability
void check_cuda() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    printf("\n=== CUDA Device Check ===\n");
    if (error == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("  Device:     %s\n", prop.name);
        printf("  Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory:     %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Status:     CUDA Available (using CPU implementation)\n\n");
    } else {
        printf("  Status:     CPU Mode (No CUDA device)\n\n");
    }
}

// Step 1: Load and test datasets
void load_and_test_dataset(const char* data_dir, Dataset** train_out, Dataset** test_out) {
    printf("\n================================================================\n");
    printf("STEP 1: Load and Preprocess CIFAR-10 Dataset\n");
    printf("================================================================\n\n");
    
    printf("Loading CIFAR-10 datasets from: %s\n", data_dir);
    
    *train_out = new Dataset(load_dataset(data_dir, true));
    printf("✓ Training dataset loaded: %d samples\n", (*train_out)->n);
    
    *test_out = new Dataset(load_dataset(data_dir, false));
    printf("✓ Test dataset loaded: %d samples\n", (*test_out)->n);
    printf("✓ Total datasets loaded: %d samples\n\n", (*train_out)->n + (*test_out)->n);
    
    if (IS_TEST_PHASE_1) {
        printf("Running verification tests:\n");
        test_sample_count(**train_out, NUM_TRAIN_SAMPLES, "Training");
        test_sample_count(**test_out, NUM_TEST_SAMPLES, "Test");
        test_normalization(**train_out, "Training");
        test_normalization(**test_out, "Test");
        printf("\n");
    }
}

// Step 2: Train autoencoder
float train_autoencoder(Cpu_Autoencoder& autoencoder, const Dataset& train_dataset,
                       int n_epoch, int batch_size, float learning_rate) {
    printf("\n================================================================\n");
    printf("STEP 2: Train Autoencoder (Unsupervised)\n");
    printf("================================================================\n\n");
    
    printf("Training Configuration:\n");
    printf("  Training samples:  %d\n", train_dataset.n);
    printf("  Epochs:            %d\n", n_epoch);
    printf("  Batch size:        %d\n", batch_size);
    printf("  Learning rate:     %.6f\n", learning_rate);
    printf("  Architecture:      32x32x3 -> 8x8x128 (8192 features)\n");
    printf("\n");
    
    float initial_loss = autoencoder.eval(train_dataset);
    printf("Initial loss (before training): %.4f\n\n", initial_loss);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    autoencoder.fit(train_dataset, n_epoch, batch_size, learning_rate, true, 0, "./model");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    float final_loss = autoencoder.eval(train_dataset);
    
    printf("\nTraining completed in %ld seconds\n", duration.count());
    printf("Final loss (after training): %.4f\n", final_loss);
    printf("Loss reduction: %.2f%%\n\n", 100.0f * (1 - final_loss/initial_loss));
    
    if (IS_TEST_PHASE_1) {
        printf("Running verification tests:\n");
        test_loss_decrease(initial_loss, final_loss);
        printf("\n");
    }
    
    return final_loss;
}

// Step 3: Extract features
void extract_features(Cpu_Autoencoder& autoencoder, 
                     const Dataset& train_dataset,
                     const Dataset& test_dataset) {
    printf("\n================================================================\n");
    printf("STEP 3: Extract Features\n");
    printf("================================================================\n\n");
    
    printf("Encoding datasets to feature space...\n\n");
    
    printf("Encoding training dataset (%d images)...\n", train_dataset.n);
    auto start_train = std::chrono::high_resolution_clock::now();
    Dataset train_features = autoencoder.encode(train_dataset);
    auto end_train = std::chrono::high_resolution_clock::now();
    auto duration_train = std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train);
    
    printf("✓ Training features extracted in %ld ms\n", duration_train.count());
    printf("  Shape: (%d, %d, %d, %d) = (%d, %d)\n\n",
           train_features.n, train_features.width, train_features.height, train_features.depth,
           train_features.n, train_features.width * train_features.height * train_features.depth);
    
    printf("Encoding test dataset (%d images)...\n", test_dataset.n);
    auto start_test = std::chrono::high_resolution_clock::now();
    Dataset test_features = autoencoder.encode(test_dataset);
    auto end_test = std::chrono::high_resolution_clock::now();
    auto duration_test = std::chrono::duration_cast<std::chrono::milliseconds>(end_test - start_test);
    
    printf("✓ Test features extracted in %ld ms\n", duration_test.count());
    printf("  Shape: (%d, %d, %d, %d) = (%d, %d)\n\n",
           test_features.n, test_features.width, test_features.height, test_features.depth,
           test_features.n, test_features.width * test_features.height * test_features.depth);
    
    if (IS_TEST_PHASE_1) {
        printf("Running verification tests:\n");
        test_encoded_dimensions(train_features, 50000, 8192, "Training features");
        test_encoded_dimensions(test_features, 10000, 8192, "Test features");
        printf("\n");
    }
    
        // Minimal output only; detailed checklist removed to reduce verbosity
}

// Main function
int main(int argc, char** argv) {
    printf("\n");
    printf("================================================================\n");
    printf("  PHASE 1: Complete Pipeline Test (Google Colab)\n");
    printf("  - Step 1: Load and Preprocess CIFAR-10\n");
    printf("  - Step 2: Train Autoencoder (Unsupervised)\n");
    printf("  - Step 3: Extract Features\n");
    printf("================================================================\n");
    
    // Default parameters for Colab
    const char* data_dir = (argc > 1) ? argv[1] : "/content/cifar-10-batches-bin";
    int n_epoch = (argc > 2) ? atoi(argv[2]) : 3;
    int batch_size = (argc > 3) ? atoi(argv[3]) : 128;
    float learning_rate = (argc > 4) ? atof(argv[4]) : 0.001f;
    
    printf("\nConfiguration:\n");
    printf("  Data directory:    %s\n", data_dir);
    printf("  Epochs:            %d\n", n_epoch);
    printf("  Batch size:        %d\n", batch_size);
    printf("  Learning rate:     %.6f\n", learning_rate);
    printf("  Test mode:         %s\n", IS_TEST_PHASE_1 ? "ENABLED" : "DISABLED");
    printf("\n");
    
    check_cuda();
    
    Dataset* train_dataset = nullptr;
    Dataset* test_dataset = nullptr;
    
    try {
        // Step 1: Load and test datasets
        load_and_test_dataset(data_dir, &train_dataset, &test_dataset);
        
        // Step 2: Train autoencoder
        printf("Initializing CPU Autoencoder...\n");
        Cpu_Autoencoder autoencoder;
        printf("✓ Autoencoder initialized\n");
        
        float final_loss = train_autoencoder(autoencoder, *train_dataset, n_epoch, batch_size, learning_rate);
        
        // Step 3: Extract features
        extract_features(autoencoder, *train_dataset, *test_dataset);
        
        // Final summary
        if (IS_TEST_PHASE_1) {
            printf("\n================================================================\n");
            printf("  PHASE 1 TEST SUMMARY\n");
            printf("================================================================\n");
            printf("  Total:   %d tests\n", stats.total_tests);
            printf("  Passed:  %d tests\n", stats.passed_tests);
            printf("  Failed:  %d tests\n", stats.failed_tests);
            
            if (stats.total_tests > 0) {
                printf("  Rate:    %.1f%%\n", 100.0f * stats.passed_tests / stats.total_tests);
            }
            
            printf("================================================================\n\n");
            
            if (stats.failed_tests == 0)
                printf("Result: ALL TESTS PASSED! Phase 1 pipeline is working correctly.\n\n");
            else
                printf("Result: SOME TESTS FAILED. Please check the output above.\n\n");
        } else {
            printf("\n================================================================\n");
            printf("  PHASE 1 COMPLETED SUCCESSFULLY\n");
            printf("================================================================\n\n");
        }
        
        delete train_dataset;
        delete test_dataset;
        
        return 0;
        
    } catch (const std::exception& e) {
        printf("\n[ERROR] Exception: %s\n\n", e.what());
        if (train_dataset) delete train_dataset;
        if (test_dataset) delete test_dataset;
        return 1;
    }
}
