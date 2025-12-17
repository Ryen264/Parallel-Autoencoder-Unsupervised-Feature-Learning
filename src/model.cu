#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#include "model.h"
using namespace std;

// Constructor
SVMmodel::SVMmodel() : accuracy(0.0), isTrained(false), n_features(0) {
    // Create cuML handle
    cumlCreate(&handle);
    
    // Initialize model pointers
    n_support = 0;
    b = 0.0f;
    dual_coefs = nullptr;
    x_support = nullptr;
    support_idx = nullptr;
    n_classes = 0;
    unique_labels = nullptr;
    
    initializeParameters();
}

// Destructor
SVMmodel::~SVMmodel() {
    if (isTrained) {
        // Free device memory
        if (dual_coefs) cudaFree(dual_coefs);
        if (x_support) cudaFree(x_support);
        if (support_idx) cudaFree(support_idx);
        if (unique_labels) cudaFree(unique_labels);
    }
    cumlDestroy(handle);
}

void SVMmodel::initializeParameters() {
    // SVM parameters
    C = 10.0f;                      // Penalty term
    cache_size = 200.0f;            // Kernel cache size in MiB
    max_iter = 100;                 // Max iterations
    nochange_steps = 100;           // Steps without change before stopping
    tol = 1e-3f;                    // Tolerance
    
    // Kernel parameters (RBF kernel)
    kernel_type = RBF;
    degree = 3;
    gamma = 0.0f;                   // Auto: 1/n_features
    coef0 = 0.0f;
}

float* SVMmodel::convertToDeviceArray(const vector<vector<double>>& data, int& n_rows, int& n_cols) {
    n_rows = data.size();
    n_cols = data[0].size();
    
    // Allocate host memory and convert to column-major format
    float* h_data = new float[n_rows * n_cols];
    for (int j = 0; j < n_cols; j++) {
        for (int i = 0; i < n_rows; i++) {
            h_data[j * n_rows + i] = static_cast<float>(data[i][j]);
        }
    }
    
    // Allocate device memory and copy
    float* d_data;
    cudaMalloc(&d_data, n_rows * n_cols * sizeof(float));
    cudaMemcpy(d_data, h_data, n_rows * n_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] h_data;
    return d_data;
}

float* SVMmodel::convertToDeviceLabels(const vector<int>& labels, int n_rows) {
    float* h_labels = new float[n_rows];
    for (int i = 0; i < n_rows; i++) {
        h_labels[i] = static_cast<float>(labels[i]);
    }
    
    float* d_labels;
    cudaMalloc(&d_labels, n_rows * sizeof(float));
    cudaMemcpy(d_labels, h_labels, n_rows * sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] h_labels;
    return d_labels;
}

void SVMmodel::freeDeviceMemory(float* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void SVMmodel::train(const vector<vector<double>>& data, 
                     const vector<int>& labels) {
    if (data.empty() || labels.empty()) {
        cerr << "Error: Empty training data" << endl;
        return;
    }
    
    if (data.size() != labels.size()) {
        cerr << "Error: Data and labels size mismatch" << endl;
        return;
    }
    
    // Convert data to device arrays
    int n_rows, n_cols;
    float* d_data = convertToDeviceArray(data, n_rows, n_cols);
    float* d_labels = convertToDeviceLabels(labels, n_rows);
    n_features = n_cols;
    
    // Set auto gamma if not set
    if (gamma == 0.0f) {
        gamma = 1.0f / n_cols;
    }
    
    // Train the model using cuML C API
    cumlError_t result = cumlSpSvcFit(
        handle,
        d_data,
        n_rows,
        n_cols,
        d_labels,
        C,
        cache_size,
        max_iter,
        nochange_steps,
        tol,
        0,  // verbosity
        kernel_type,
        degree,
        gamma,
        coef0,
        &n_support,
        &b,
        &dual_coefs,
        &x_support,
        &support_idx,
        &n_classes,
        &unique_labels
    );
    
    if (result == CUML_SUCCESS) {
        isTrained = true;
        cout << "Model trained successfully" << endl;
        cout << "Number of support vectors: " << n_support << endl;
    } else {
        cerr << "Error during training: cuML error code " << result << endl;
    }
    
    // Free temporary device memory
    freeDeviceMemory(d_data);
    freeDeviceMemory(d_labels);
}

int SVMmodel::predict(const vector<double>& sample) const {
    if (!isTrained) {
        cerr << "Error: Model not trained" << endl;
        return -1;
    }
    
    // Convert single sample to column-major format
    int n_cols = sample.size();
    float* h_sample = new float[n_cols];
    for (int i = 0; i < n_cols; i++) {
        h_sample[i] = static_cast<float>(sample[i]);
    }
    
    // Copy to device
    float* d_sample;
    cudaMalloc(&d_sample, n_cols * sizeof(float));
    cudaMemcpy(d_sample, h_sample, n_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate prediction output on device
    float* d_pred;
    cudaMalloc(&d_pred, sizeof(float));
    
    // Predict using cuML C API
    cumlError_t result = cumlSpSvcPredict(
        handle,
        d_sample,
        1,
        n_features,
        kernel_type,
        degree,
        gamma,
        coef0,
        n_support,
        b,
        dual_coefs,
        x_support,
        n_classes,
        unique_labels,
        d_pred,
        200.0f,  // buffer_size
        1        // predict_class = true
    );
    
    // Copy result back to host
    float h_pred = -1;
    if (result == CUML_SUCCESS) {
        cudaMemcpy(&h_pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cerr << "Error during prediction" << endl;
    }
    
    // Cleanup
    delete[] h_sample;
    cudaFree(d_sample);
    cudaFree(d_pred);
    
    return static_cast<int>(h_pred);
}

vector<int> SVMmodel::predictBatch(const vector<vector<double>>& samples) const {
    if (!isTrained) {
        cerr << "Error: Model not trained" << endl;
        return vector<int>();
    }
    
    // Convert batch to device array
    int n_rows, n_cols;
    float* d_samples = const_cast<SVMmodel*>(this)->convertToDeviceArray(samples, n_rows, n_cols);
    
    // Allocate predictions on device
    float* d_preds;
    cudaMalloc(&d_preds, n_rows * sizeof(float));
    
    // Predict using cuML C API
    cumlError_t result = cumlSpSvcPredict(
        handle,
        d_samples,
        n_rows,
        n_features,
        kernel_type,
        degree,
        gamma,
        coef0,
        n_support,
        b,
        dual_coefs,
        x_support,
        n_classes,
        unique_labels,
        d_preds,
        200.0f,  // buffer_size
        1        // predict_class = true
    );
    
    vector<int> predictions;
    if (result == CUML_SUCCESS) {
        // Copy predictions back to host
        float* h_preds = new float[n_rows];
        cudaMemcpy(h_preds, d_preds, n_rows * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Convert to vector<int>
        predictions.resize(n_rows);
        for (int i = 0; i < n_rows; i++) {
            predictions[i] = static_cast<int>(h_preds[i]);
        }
        delete[] h_preds;
    } else {
        cerr << "Error during batch prediction" << endl;
    }
    
    // Cleanup
    const_cast<SVMmodel*>(this)->freeDeviceMemory(d_samples);
    cudaFree(d_preds);
    
    return predictions;
}

double SVMmodel::test(const vector<vector<double>>& testData, 
                      const vector<int>& testLabels) {
    if (!isTrained) {
        cerr << "Error: Model not trained" << endl;
        return 0.0;
    }
    
    vector<int> predictions = predictBatch(testData);
    
    int correct = 0;
    for (size_t i = 0; i < testLabels.size(); i++) {
        if (predictions[i] == testLabels[i]) {
            correct++;
        }
    }
    
    accuracy = (double)correct / testLabels.size();
    
    cout << "Test Accuracy: " << (accuracy * 100.0) << "%" << endl;
    
    // Print confusion matrix
    printConfusionMatrix(predictions, testLabels);
    
    return accuracy;
}

void SVMmodel::printConfusionMatrix(const vector<int>& predicted, 
                                   const vector<int>& actual, int numClasses) {
    vector<vector<int>> confusionMatrix(numClasses, vector<int>(numClasses, 0));
    
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] >= 0 && actual[i] < numClasses && 
            predicted[i] >= 0 && predicted[i] < numClasses) {
            confusionMatrix[actual[i]][predicted[i]]++;
        }
    }
    
    cout << "\nConfusion Matrix:" << endl;
    cout << "Actual \\ Predicted | ";
    for (int i = 0; i < numClasses; i++) {
        cout << i << " ";
    }
    cout << endl;
    cout << "--------------------" << endl;
    
    for (int i = 0; i < numClasses; i++) {
        cout << i << " | ";
        for (int j = 0; j < numClasses; j++) {
            cout << confusionMatrix[i][j] << " ";
        }
        cout << endl;
    }
}

bool SVMmodel::save(const string& modelPath) const {
    if (!isTrained) {
        cerr << "Error: Model not trained or invalid" << endl;
        return false;
    }
    
    // TODO: Implement custom model serialization for cuML
    // cuML doesn't provide built-in save/load like libsvm
    cerr << "Warning: Model save/load not yet implemented for cuML" << endl;
    return false;
}

bool SVMmodel::load(const string& modelPath) {
    // TODO: Implement custom model deserialization for cuML
    cerr << "Warning: Model save/load not yet implemented for cuML" << endl;
    return false;
}

double SVMmodel::getAccuracy() const {
    return accuracy;
}

bool SVMmodel::getIsTrained() const {
    return isTrained;
}

void SVMmodel::printModelInfo() const {
    if (!isTrained) {
        cout << "Model is not trained" << endl;
        return;
    }
    
    cout << "\n=== cuML SVM Model Information ===" << endl;
    cout << "Number of support vectors: " << n_support << endl;
    cout << "Number of classes: " << n_classes << endl;
    cout << "Number of features: " << n_features << endl;
    cout << "Kernel type: RBF" << endl;
    cout << "C parameter: " << C << endl;
    cout << "Gamma: " << gamma << endl;
    cout << "Accuracy: " << (accuracy * 100.0) << "%" << endl;
    cout << "============================\n" << endl;
}