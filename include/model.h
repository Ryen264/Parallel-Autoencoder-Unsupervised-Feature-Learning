#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <vector>
#include <string>
#include <libsvm/svm.h>
//!git clone https://github.com/cjlin1/libsvm.git
//#include "/content/libsvm/svm.h"

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
    svm_model* model;
    svm_parameter parameters;
    double accuracy;
    bool isTrained;
    
    // Helper methods
    svm_problem* createProblem(const std::vector<std::vector<double>>& data, 
                              const std::vector<int>& labels);
    void freeProblem(svm_problem* problem);
    void initializeParameters();

public:
    // Constructor and Destructor
    SVMmodel();
    ~SVMmodel();

    // Training
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);
    
    // Prediction
    int predict(const std::vector<double>& sample) const;
    std::vector<int> predictBatch(const std::vector<std::vector<double>>& samples) const;
    
    // Testing
    double test(const std::vector<std::vector<double>>& testData, 
               const std::vector<int>& testLabels);
    void printConfusionMatrix(const std::vector<int>& predicted, 
                             const std::vector<int>& actual, int numClasses = 10);
    
    // Model persistence
    bool save(const std::string& modelPath) const;
    bool load(const std::string& modelPath);
    
    // Utility methods
    double getAccuracy() const;
    bool isTrained() const;
    void printModelInfo() const;
};

#endif