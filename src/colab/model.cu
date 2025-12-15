#include <iostream>
#include <cstdlib>

#include "model.h"
using namespace std;

// Constructor
SVMmodel::SVMmodel() : accuracy(0.0), isTrained(false) {
    initializeParameters();
}

// Destructor
SVMmodel::~SVMmodel() {
    // cuML handles cleanup automatically
}

void SVMmodel::initializeParameters() {
    parameters.C = 10.0;
    parameters.kernel = cuml::SVM::RBF;
    parameters.degree = 3;
    parameters.gamma = 0.0; // Auto
    parameters.coef0 = 0.0;
    parameters.tol = 1e-3;
    parameters.max_iter = 10000;
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
    
    int numSamples = data.size();
    int numFeatures = data[0].size();
    
    // Flatten data
    vector<double> X(numSamples * numFeatures);
    for(int i = 0; i < numSamples; i++) {
        for(int j = 0; j < numFeatures; j++) {
            X[i * numFeatures + j] = data[i][j];
        }
    }
    
    vector<double> y(labels.begin(), labels.end());
    
    // Train model
    model.fit(X.data(), numSamples, numFeatures, y.data(), parameters);
    
    isTrained = true;
    cout << "Model trained successfully" << endl;
}

int SVMmodel::predict(const vector<double>& sample) const {
    if (!isTrained) {
        cerr << "Error: Model not trained" << endl;
        return -1;
    }
    
    double prediction;
    model.predict(sample.data(), 1, sample.size(), &prediction);
    
    return (int)prediction;
}

vector<int> SVMmodel::predictBatch(const vector<vector<double>>& samples) const {
    vector<int> predictions;
    for (const auto& sample : samples) {
        predictions.push_back(predict(sample));
    }
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
    
    model.save(modelPath);
    
    cout << "Model saved to " << modelPath << endl;
    return true;
}

bool SVMmodel::load(const string& modelPath) {
    model.load(modelPath);
    
    isTrained = true;
    cout << "Model loaded from " << modelPath << endl;
    return true;
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
    
    cout << "\n=== SVM Model Information ===" << endl;
    cout << "Kernel type: RBF" << endl;
    cout << "C parameter: " << parameters.C << endl;
    cout << "Accuracy: " << (accuracy * 100.0) << "%" << endl;
    cout << "============================\n" << endl;
}