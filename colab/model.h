#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include "constants.h"
#include "libsvm/svm.h"
#include "macro.h"
#include "progress_bar.h"
#include "timer.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <set>
// mkdir/stat for cross-platform directory handling used in save_evaluation()
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif
using namespace std;

// Forward declaration of libsvm structures
struct svm_model;
struct svm_node;

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

    // LIBSVM model
    svm_model* svm_model;
    
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
    float   bias;
    int     n_classes;
    
    // Helper methods for data conversion
    float*  convertToDeviceArray(const vector<vector<double>>& data, int& n_rows, int& n_cols);
    float*  convertToDeviceLabels(const vector<int>& labels, int n_rows);
    void    freeDeviceMemory(float* ptr);

public:
    // Constructor and Destructor
    SVMmodel();
    SVMmodel(float C, string kernel_type, string gamma_type);
    SVMmodel(float C, string kernel_type, string gamma_type, float tolerance, float cache_size, int max_iter, int nochange_steps);
    
    // Copy constructor
    SVMmodel(const SVMmodel &other);
    
    // Move constructor
    SVMmodel(SVMmodel &&other) noexcept;
    
    ~SVMmodel();

    // Copy assignment operator overload
    SVMmodel &operator=(const SVMmodel &other);

    // Move assignment operator overload
    SVMmodel &operator=(SVMmodel &&other) noexcept;

    // Training
    void train(const vector<vector<double>>& data, const vector<int>& labels);
    
    // Prediction
    vector<int> predict(const vector<vector<double>>& samples) const;
    double calculateAccuracy(const vector<int>& predicted, const vector<int>& actual, int numClasses = 10);
    vector<vector<int>> calculateClassificationReport(const vector<int>& predicted, const vector<int>& actual, int numClasses = 10);
    vector<vector<int>> calculateConfusionMatrix(const vector<int>& predicted, const vector<int>& actual, int numClasses = 10);

    // Testing
    void printClassificationReport(const vector<vector<int>>& classificationReport);
    void printConfusionMatrix(const vector<vector<int>>& confusionMatrix);

    // Model persistence
    bool save(const string& modelPath) const;
    bool load(const string& modelPath);
    
    // Evaluation persistence
    bool save_evaluation(double accuracy, const vector<vector<int>>& class_report, 
                        const vector<vector<int>>& conf_matrix, const string& eval_path) const;
    
    // Utility methods
    bool getIsTrained() const;
    void printModelInfo() const;
};

#endif