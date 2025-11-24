/*
 * CIFAR-10 Data Loader Test Program (Google Colab Version)
 * 
 * Tests loading and preprocessing of CIFAR-10 dataset:
 * - Verifies 50,000 training images loaded correctly
 * - Verifies 10,000 test images loaded correctly  
 * - Verifies preprocessing: normalization to [0,1]
 * - Checks data integrity, dimensions, and statistics
 * 
 * Default data path for Google Colab: /content/cifar-10-batches-bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "data_loader.h"
#include "constants.h"

// Global flag to control test mode
bool IS_TEST_DATA_LOADER = true;  // Set to true by default for Colab testing

// Test statistics structure
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
} TestStats;

TestStats stats = {0, 0, 0};

// Print colored test result
void print_test(const char* name, bool passed) {
    if (!IS_TEST_DATA_LOADER) return;
    
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

// Test: Verify image dimensions
bool test_dimensions(const Dataset& dataset, int exp_width, int exp_depth, const char* name) {
    bool passed = (dataset.width == exp_width && dataset.depth == exp_depth);
    char msg[256];
    snprintf(msg, sizeof(msg), "%s dimensions: %dx%dx%d (expected %dx%dx%d)", 
             name, dataset.width, dataset.width, dataset.depth,
             exp_width, exp_width, exp_depth);
    print_test(msg, passed);
    return passed;
}

// Test: Verify normalization to [0,1]
bool test_normalization(const Dataset& dataset, const char* name) {
    float* data = dataset.get_data();
    int total_size = dataset.n * dataset.width * dataset.width * dataset.depth;
    
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
    
    printf("       Min: %.6f, Max: %.6f\n", min_val, max_val);
    return passed;
}

// Test: Verify labels are in valid range [0-9]
bool test_labels(const Dataset& dataset, const char* name) {
    int* labels = dataset.get_labels();
    bool passed = true;
    int invalid_count = 0;
    
    for (int i = 0; i < dataset.n; i++) {
        if (labels[i] < 0 || labels[i] >= NUM_CLASSES) {
            passed = false;
            invalid_count++;
        }
    }
    
    char msg[256];
    if (passed) {
        snprintf(msg, sizeof(msg), "%s labels valid [0-%d]", name, NUM_CLASSES-1);
    } else {
        snprintf(msg, sizeof(msg), "%s labels valid [0-%d] (%d invalid found)", 
                 name, NUM_CLASSES-1, invalid_count);
    }
    print_test(msg, passed);
    return passed;
}

// Test: Check data variance (not all zeros or constant)
bool test_data_variance(const Dataset& dataset, const char* name) {
    float* data = dataset.get_data();
    float first = data[0];
    bool has_variance = false;
    
    int check_size = (dataset.n * IMAGE_SIZE > 10000) ? 10000 : dataset.n * IMAGE_SIZE;
    for (int i = 1; i < check_size; i++) {
        if (fabs(data[i] - first) > 0.001f) {
            has_variance = true;
            break;
        }
    }
    
    char msg[256];
    snprintf(msg, sizeof(msg), "%s has data variance (not constant)", name);
    print_test(msg, has_variance);
    return has_variance;
}

// Calculate and print dataset statistics
void print_statistics(const Dataset& dataset, const char* name) {
    if (!IS_TEST_DATA_LOADER) return;
    
    float* data = dataset.get_data();
    int* labels = dataset.get_labels();
    int total_size = dataset.n * dataset.width * dataset.width * dataset.depth;
    
    // Calculate mean and std dev
    double sum = 0.0;
    for (int i = 0; i < total_size; i++) {
        sum += data[i];
    }
    float mean = sum / total_size;
    
    double var_sum = 0.0;
    for (int i = 0; i < total_size; i++) {
        double diff = data[i] - mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrt(var_sum / total_size);
    
    // Count labels
    int label_counts[NUM_CLASSES] = {0};
    for (int i = 0; i < dataset.n; i++) {
        if (labels[i] >= 0 && labels[i] < NUM_CLASSES) {
            label_counts[labels[i]]++;
        }
    }
    
    printf("\n  === %s Statistics ===\n", name);
    printf("  Samples:     %d\n", dataset.n);
    printf("  Dimensions:  %dx%dx%d\n", dataset.width, dataset.width, dataset.depth);
    printf("  Mean:        %.4f\n", mean);
    printf("  Std Dev:     %.4f\n", std_dev);
    printf("  Label Distribution:\n");
    
    const char* classes[] = {"airplane", "automobile", "bird", "cat", "deer",
                             "dog", "frog", "horse", "ship", "truck"};
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("    %d (%-10s): %5d (%.1f%%)\n", 
               i, classes[i], label_counts[i], 
               100.0f * label_counts[i] / dataset.n);
    }
    printf("\n");
}

// Test: Verify class balance (each class ~10%)
bool test_class_balance(const Dataset& dataset, const char* name) {
    int* labels = dataset.get_labels();
    int counts[NUM_CLASSES] = {0};
    
    for (int i = 0; i < dataset.n; i++) {
        if (labels[i] >= 0 && labels[i] < NUM_CLASSES) {
            counts[labels[i]]++;
        }
    }
    
    bool balanced = true;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float pct = 100.0f * counts[i] / dataset.n;
        if (pct < 8.0f || pct > 12.0f) {
            balanced = false;
            break;
        }
    }
    
    char msg[256];
    snprintf(msg, sizeof(msg), "%s has balanced classes (~10%% each)", name);
    print_test(msg, balanced);
    return balanced;
}

// Test: Verify RGB structure of images
bool test_rgb_structure(const Dataset& dataset, const char* name) {
    float* data = dataset.get_data();
    bool valid = true;
    
    // Check first few images
    int check_count = (dataset.n > 5) ? 5 : dataset.n;
    for (int img = 0; img < check_count; img++) {
        int offset = img * IMAGE_SIZE;
        int pix_per_ch = IMAGE_WIDTH * IMAGE_WIDTH;
        
        float r_sum = 0, g_sum = 0, b_sum = 0;
        for (int i = 0; i < pix_per_ch; i++) {
            r_sum += data[offset + i];
            g_sum += data[offset + pix_per_ch + i];
            b_sum += data[offset + 2 * pix_per_ch + i];
        }
        
        float r_mean = r_sum / pix_per_ch;
        float g_mean = g_sum / pix_per_ch;
        float b_mean = b_sum / pix_per_ch;
        
        // Check for suspicious all-black or all-white
        if ((r_mean < 0.01f && g_mean < 0.01f && b_mean < 0.01f) ||
            (r_mean > 0.99f && g_mean > 0.99f && b_mean > 0.99f)) {
            valid = false;
            break;
        }
    }
    
    char msg[256];
    snprintf(msg, sizeof(msg), "%s has valid RGB structure", name);
    print_test(msg, valid);
    return valid;
}

// Check CUDA availability
void check_cuda() {
    if (!IS_TEST_DATA_LOADER) return;
    
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    printf("\n=== CUDA Device Check ===\n");
    if (error == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("  Device:     %s\n", prop.name);
        printf("  Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory:     %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Status:     CUDA Available\n\n");
    } else {
        printf("  Status:     CPU Mode (No CUDA device)\n\n");
    }
}

// Load datasets without testing
void load_datasets(const char* data_dir, Dataset** train_out, Dataset** test_out) {
    printf("Loading CIFAR-10 datasets from: %s\n", data_dir);
    
    // Load training dataset
    *train_out = new Dataset(load_dataset(data_dir, true));
    printf("✓ Training dataset loaded: %d samples\n", (*train_out)->n);
    
    // Load test dataset
    *test_out = new Dataset(load_dataset(data_dir, false));
    printf("✓ Test dataset loaded: %d samples\n", (*test_out)->n);
    
    printf("✓ Total datasets loaded: %d samples\n", (*train_out)->n + (*test_out)->n);
}

// Run comprehensive tests on loaded datasets
void run_tests(const Dataset& train_dataset, const Dataset& test_dataset) {
    bool all_passed = true;
    
    // ========================================================================
    // TEST 1: Training dataset verification
    // ========================================================================
    printf("\n================================================================\n");
    printf("TEST 1: Training Dataset (50,000 images)\n");
    printf("================================================================\n\n");
    
    printf("Running verification tests:\n");
    all_passed &= test_sample_count(train_dataset, NUM_TRAIN_SAMPLES, "Training");
    all_passed &= test_dimensions(train_dataset, IMAGE_WIDTH, IMAGE_DEPTH, "Training");
    all_passed &= test_normalization(train_dataset, "Training");
    all_passed &= test_labels(train_dataset, "Training");
    all_passed &= test_data_variance(train_dataset, "Training");
    all_passed &= test_rgb_structure(train_dataset, "Training");
    all_passed &= test_class_balance(train_dataset, "Training");
    
    print_statistics(train_dataset, "Training Dataset");
    
    // ========================================================================
    // TEST 2: Test dataset verification
    // ========================================================================
    printf("================================================================\n");
    printf("TEST 2: Test Dataset (10,000 images)\n");
    printf("================================================================\n\n");
    
    printf("Running verification tests:\n");
    all_passed &= test_sample_count(test_dataset, NUM_TEST_SAMPLES, "Test");
    all_passed &= test_dimensions(test_dataset, IMAGE_WIDTH, IMAGE_DEPTH, "Test");
    all_passed &= test_normalization(test_dataset, "Test");
    all_passed &= test_labels(test_dataset, "Test");
    all_passed &= test_data_variance(test_dataset, "Test");
    all_passed &= test_rgb_structure(test_dataset, "Test");
    all_passed &= test_class_balance(test_dataset, "Test");
    
    print_statistics(test_dataset, "Test Dataset");
    
    // ========================================================================
    // TEST 3: Dataset integrity
    // ========================================================================
    printf("================================================================\n");
    printf("TEST 3: Dataset Integrity\n");
    printf("================================================================\n\n");
    
    float* train_data = train_dataset.get_data();
    float* test_data = test_dataset.get_data();
    
    bool different = false;
    for (int i = 0; i < IMAGE_SIZE && i < IMAGE_SIZE; i++) {
        if (fabs(train_data[i] - test_data[i]) > 0.001f) {
            different = true;
            break;
        }
    }
    
    print_test("Training and test datasets are different", different);
    all_passed &= different;
    
    int total = train_dataset.n + test_dataset.n;
    bool correct_total = (total == 60000);
    char msg[256];
    snprintf(msg, sizeof(msg), "Total samples = 60,000 (actual: %d)", total);
    print_test(msg, correct_total);
    all_passed &= correct_total;
    
    // ========================================================================
    // Final Summary
    // ========================================================================
    printf("\n");
    printf("================================================================\n");
    printf("  TEST SUMMARY\n");
    printf("================================================================\n");
    printf("  Total:   %d tests\n", stats.total_tests);
    printf("  Passed:  %d tests\n", stats.passed_tests);
    printf("  Failed:  %d tests\n", stats.failed_tests);
    printf("  Rate:    %.1f%%\n", 100.0f * stats.passed_tests / stats.total_tests);
    printf("================================================================\n\n");
    
    // Verification checklist
    printf("VERIFICATION CHECKLIST:\n");
    printf("  %s 50,000 training images loaded\n", all_passed ? "[✓]" : "[✗]");
    printf("  %s 10,000 test images loaded\n", all_passed ? "[✓]" : "[✗]");
    printf("  %s Preprocessing: normalized to [0,1]\n", all_passed ? "[✓]" : "[✗]");
    printf("\n");
    
    if (all_passed && stats.failed_tests == 0) {
        printf("Result: ALL TESTS PASSED! Data loader is working correctly.\n\n");
    } else {
        printf("Result: SOME TESTS FAILED. Please check the output above.\n\n");
    }
}

// Main function
int main(int argc, char** argv) {
    printf("\n");
    printf("================================================================\n");
    printf("  CIFAR-10 Data Loader (Google Colab Version)\n");
    printf("================================================================\n");
    
    // Default data directory for Google Colab
    const char* data_dir = (argc > 1) ? argv[1] : "/content/cifar-10-batches-bin";
    printf("Data directory: %s\n", data_dir);
    
    // Check CUDA
    check_cuda();
    
    Dataset* train_dataset = nullptr;
    Dataset* test_dataset = nullptr;
    
    try {
        // Load datasets (always runs)
        load_datasets(data_dir, &train_dataset, &test_dataset);
        
        // Run tests only if IS_TEST_DATA_LOADER is true
        if (IS_TEST_DATA_LOADER) {
            run_tests(*train_dataset, *test_dataset);
        } else {
            printf("\nDatasets loaded successfully. Test mode disabled (IS_TEST_DATA_LOADER = false).\n");
            printf("Set IS_TEST_DATA_LOADER = true to run comprehensive tests.\n\n");
        }
        
        // Cleanup
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
