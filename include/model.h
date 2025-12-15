#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <vector>
#include <string>

// #include <libsvm/svm.h>
// !git clone https://github.com/cjlin1/libsvm.git
#include "/content/libsvm/svm.h"
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
    svm_model* model;
    svm_parameter parameters;
    double accuracy;
    bool isTrained;
    
    // Helper methods
    svm_problem* createProblem(const vector<vector<double>>& data, 
                              const vector<int>& labels);
    void freeProblem(svm_problem* problem);
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