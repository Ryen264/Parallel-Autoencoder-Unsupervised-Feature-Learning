#include <iostream>
#include <cstdlib>
#include "model.h"

// Constructor
SVMmodel::SVMmodel() : model(nullptr), accuracy(0.0), isTrained(false) {
    initializeParameters();
}

// Destructor
SVMmodel::~SVMmodel() {
    if (model != nullptr) {
        svm_free_model_content(model);
        free(model);
    }
}

void SVMmodel::initializeParameters() {
    parameters.svm_type = C_SVC;           // C-classification
    parameters.kernel_type = RBF;          // RBF kernel
    parameters.degree = 3;
    parameters.gamma = 0;                  // Auto: 1/num_features
    parameters.coef0 = 0;
    parameters.C = 10.0;                   // Hyperparameter C
    parameters.nu = 0.5;
    parameters.p = 0.1;
    parameters.cache_size = 100;
    parameters.eps = 1e-3;
    parameters.shrinking = 1;
    parameters.probability = 1;            // Enable probability estimates
    parameters.nr_weight = 0;
    parameters.weight_label = nullptr;
    parameters.weight = nullptr;
}

svm_problem* SVMmodel::createProblem(const vector<vector<double>>& data, 
                                     const vector<int>& labels) {
    int numSamples = data.size();
    int numFeatures = data[0].size();
    
    svm_problem* problem = (svm_problem*)malloc(sizeof(svm_problem));
    problem->l = numSamples;
    problem->y = (double*)malloc(numSamples * sizeof(double));
    problem->x = (svm_node**)malloc(numSamples * sizeof(svm_node*));
    
    // Copy labels
    for (int i = 0; i < numSamples; i++) {
        problem->y[i] = (double)labels[i];
    }
    
    // Copy features
    for (int i = 0; i < numSamples; i++) {
        problem->x[i] = (svm_node*)malloc((numFeatures + 1) * sizeof(svm_node));
        for (int j = 0; j < numFeatures; j++) {
            problem->x[i][j].index = j + 1;  // libsvm uses 1-based indexing
            problem->x[i][j].value = data[i][j];
        }
        problem->x[i][numFeatures].index = -1;  // End marker
    }
    
    return problem;
}

void SVMmodel::freeProblem(svm_problem* problem) {
    if (problem != nullptr) {
        for (int i = 0; i < problem->l; i++) {
            free(problem->x[i]);
        }
        free(problem->x);
        free(problem->y);
        free(problem);
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
    
    // Create problem structure
    svm_problem* problem = createProblem(data, labels);
    
    // Check parameters
    const char* errMsg = svm_check_parameter(problem, &parameters);
    if (errMsg != nullptr) {
        cerr << "Error: " << errMsg << endl;
        freeProblem(problem);
        return;
    }
    
    // Train model
    svm_set_print_string_function([](const char*) {}); // Suppress output
    model = svm_train(problem, &parameters);
    
    freeProblem(problem);
    isTrained = true;
    cout << "Model trained successfully" << endl;
}

int SVMmodel::predict(const vector<double>& sample) const {
    if (!isTrained || model == nullptr) {
        cerr << "Error: Model not trained" << endl;
        return -1;
    }
    
    // Create feature node
    int numFeatures = sample.size();
    svm_node* x = (svm_node*)malloc((numFeatures + 1) * sizeof(svm_node));
    for (int i = 0; i < numFeatures; i++) {
        x[i].index = i + 1;
        x[i].value = sample[i];
    }
    x[numFeatures].index = -1;
    
    // Predict
    double prediction = svm_predict(model, x);
    
    free(x);
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
    if (!isTrained || model == nullptr) {
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
    if (!isTrained || model == nullptr) {
        cerr << "Error: Model not trained or invalid" << endl;
        return false;
    }
    
    if (svm_save_model(modelPath.c_str(), model) != 0) {
        cerr << "Error: Failed to save model to " << modelPath << endl;
        return false;
    }
    
    cout << "Model saved to " << modelPath << endl;
    return true;
}

bool SVMmodel::load(const string& modelPath) {
    model = svm_load_model(modelPath.c_str());
    if (model == nullptr) {
        cerr << "Error: Failed to load model from " << modelPath << endl;
        return false;
    }
    
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
    if (!isTrained || model == nullptr) {
        cout << "Model is not trained" << endl;
        return;
    }
    
    cout << "\n=== SVM Model Information ===" << endl;
    cout << "Number of support vectors: " << model->l << endl;
    cout << "Number of classes: " << model->nr_class << endl;
    cout << "Number of features: " << (model->SV[0] != nullptr ? "computed" : "unknown") << endl;
    cout << "Kernel type: RBF" << endl;
    cout << "C parameter: " << parameters.C << endl;
    cout << "Accuracy: " << (accuracy * 100.0) << "%" << endl;
    cout << "============================\n" << endl;
}