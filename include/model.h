#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <vector>
#include <string>

// cuML SVM C API header
#include <cuml/svm/svm_api.h>
#include <cuml/cuml_api.h>

using namespace std;

class SVMmodel {
/*
Train SVM (Library)
+ Input: train_features + labels  
+ Kernel: RBF (Radial Basis Function)  
+ Hyperparameters: C=10, gamma=auto  
+ Output: trained SVM model  

Evaluate
+ Predict on test_features using SVM  
+ Calculate accuracy, confusion matrix  
+ Compare with baseline methods
*/
private:
    cumlHandle_t handle;
    
    // Model parameters
    int n_support;
    float b;
    float* dual_coefs;
    float* x_support;
    int* support_idx;
    int n_classes;
    float* unique_labels;
    
    // Training parameters
    float C;
    float cache_size;
    int max_iter;
    int nochange_steps;
    float tol;
    cumlSvmKernelType kernel_type;
    int degree;
    float gamma;
    float coef0;
    
    double accuracy;
    bool isTrained;
    int n_features;
    
    // Helper methods for data conversion
    float* convertToDeviceArray(const vector<vector<double>>& data, int& n_rows, int& n_cols);
    float* convertToDeviceLabels(const vector<int>& labels, int n_rows);
    void freeDeviceMemory(float* ptr);
    void initializeParameters();

public:
    // Constructor and Destructor
    SVMmodel();
    ~SVMmodel();

    // Training
    void train(const vector<vector<double>>& data, const vector<int>& labels);
    
    // Prediction
    int predict(const vector<double>& sample) const;
    vector<int> predictBatch(const vector<vector<double>>& samples) const;
    
    // Testing
    double test(const vector<vector<double>>& testData, 
               const vector<int>& testLabels);
    void printConfusionMatrix(const vector<int>& predicted, 
                             const vector<int>& actual, int numClasses = 10);
    
    // Model persistence
    bool save(const string& modelPath) const;
    bool load(const string& modelPath);
    
    // Utility methods
    double getAccuracy() const;
    bool getIsTrained() const;
    void printModelInfo() const;
};

#endif