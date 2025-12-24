#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include "constants.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>   // CUDA runtime for cudaStream_t
#include <cuml/svm/svc.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>
using namespace std;

class SVMmodel {
/*
Train SVM (Library)
+ Input: train_features + labels  
+ Kernel: RBF (Radial Basis Function)  
+ Hyperparameters: C=10, gamma=(float)auto
+ Output: trained SVM model  

Evaluate
+ Predict on test_features using SVM  
+ Calculate accuracy, confusion matrix  
+ Compare with baseline methods
*/
private:
    bool    is_trained;
    int     n_features;

    raft::handle_t handle;        // RAFT handle
    ML::SVM::SVC<float> svm_model;          // cuML ML::SVM model
    
    // Training parameters
    float   C;
    string  kernel_type;
    int     kernel;
    string  gamma_type;
    float   gamma;

    float   tolerance;
    float   cache_size;
    int     max_iter;
    int     nochange_steps;
    int     verbosity;

    // Model parameters
    int     n_support;
    float   b;
    float*  dual_coefs;
    float*  x_support;
    int*    support_idx;
    int     n_classes;
    float*  unique_labels;
    
    // Helper methods for data conversion
    float* convertToDeviceArray(const vector<vector<double>>& data, int& n_rows, int& n_cols);
    float* convertToDeviceLabels(const vector<int>& labels, int n_rows);
    void freeDeviceMemory(float* ptr);

public:
    // Constructor and Destructor
    SVMmodel();
    SVMmodel(float C, string kernel_type, string gamma_type);
    SVMmodel(float C, string kernel_type, string gamma_type, float tolerance, float cache_size, int max_iter, int nochange_steps);
    ~SVMmodel();

    // Training
    void train(const vector<vector<double>>& data, const vector<int>& labels);
    
    // Prediction
    vector<int> predict(const vector<vector<double>>& samples) const;
    double calculateAccuracy(const vector<int>& predicted, const vector<int>& actual, int numClasses = 10);
    vector<vector<int>> calculateConfusionMatrix(const vector<int>& predicted, const vector<int>& actual, int numClasses = 10);

    // Testing
    void printClassificationReport(const vector<vector<int>>& classificationReport);
    void printConfusionMatrix(const vector<vector<int>>& confusionMatrix);

    // Model persistence
    bool save(const string& modelPath) const;
    bool load(const string& modelPath);
    
    // Utility methods
    bool getIsTrained() const;
    void printModelInfo() const;
};

#endif