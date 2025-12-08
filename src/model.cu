#include <iostream>
#include <cstdlib>
#include "../include/model.h"

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

svm_problem* SVMmodel::createProblem(const std::vector<std::vector<double>>& data, 
                                     const std::vector<int>& labels) {
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

void SVMmodel::train(const std::vector<std::vector<double>>& data, 
                     const std::vector<int>& labels) {
    if (data.empty() || labels.empty()) {
        std::cerr << "Error: Empty training data" << std::endl;
        return;
    }
    
    if (data.size() != labels.size()) {
        std::cerr << "Error: Data and labels size mismatch" << std::endl;
        return;
    }
    
    // Create problem structure
    svm_problem* problem = createProblem(data, labels);
    
    // Check parameters
    const char* errMsg = svm_check_parameter(problem, &parameters);
    if (errMsg != nullptr) {
        std::cerr << "Error: " << errMsg << std::endl;
        freeProblem(problem);
        return;
    }
    
    // Train model
    svm_set_print_string_function([](const char*) {}); // Suppress output
    model = svm_train(problem, &parameters);
    
    freeProblem(problem);
    isTrained = true;
    std::cout << "Model trained successfully" << std::endl;
}

int SVMmodel::predict(const std::vector<double>& sample) const {
    if (!isTrained || model == nullptr) {
        std::cerr << "Error: Model not trained" << std::endl;
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

std::vector<int> SVMmodel::predictBatch(const std::vector<std::vector<double>>& samples) const {
    std::vector<int> predictions;
    for (const auto& sample : samples) {
        predictions.push_back(predict(sample));
    }
    return predictions;
}

double SVMmodel::test(const std::vector<std::vector<double>>& testData, 
                      const std::vector<int>& testLabels) {
    if (!isTrained || model == nullptr) {
        std::cerr << "Error: Model not trained" << std::endl;
        return 0.0;
    }
    
    std::vector<int> predictions = predictBatch(testData);
    
    int correct = 0;
    for (size_t i = 0; i < testLabels.size(); i++) {
        if (predictions[i] == testLabels[i]) {
            correct++;
        }
    }
    
    accuracy = (double)correct / testLabels.size();
    
    std::cout << "Test Accuracy: " << (accuracy * 100.0) << "%" << std::endl;
    
    // Print confusion matrix
    printConfusionMatrix(predictions, testLabels);
    
    return accuracy;
}

void SVMmodel::printConfusionMatrix(const std::vector<int>& predicted, 
                                   const std::vector<int>& actual, int numClasses) {
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] >= 0 && actual[i] < numClasses && 
            predicted[i] >= 0 && predicted[i] < numClasses) {
            confusionMatrix[actual[i]][predicted[i]]++;
        }
    }
    
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "Actual \\ Predicted | ";
    for (int i = 0; i < numClasses; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "--------------------" << std::endl;
    
    for (int i = 0; i < numClasses; i++) {
        std::cout << i << " | ";
        for (int j = 0; j < numClasses; j++) {
            std::cout << confusionMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

bool SVMmodel::save(const std::string& modelPath) const {
    if (!isTrained || model == nullptr) {
        std::cerr << "Error: Model not trained or invalid" << std::endl;
        return false;
    }
    
    if (svm_save_model(modelPath.c_str(), model) != 0) {
        std::cerr << "Error: Failed to save model to " << modelPath << std::endl;
        return false;
    }
    
    std::cout << "Model saved to " << modelPath << std::endl;
    return true;
}

bool SVMmodel::load(const std::string& modelPath) {
    model = svm_load_model(modelPath.c_str());
    if (model == nullptr) {
        std::cerr << "Error: Failed to load model from " << modelPath << std::endl;
        return false;
    }
    
    isTrained = true;
    std::cout << "Model loaded from " << modelPath << std::endl;
    return true;
}

double SVMmodel::getAccuracy() const {
    return accuracy;
}

bool SVMmodel::isTrained() const {
    return isTrained;
}

void SVMmodel::printModelInfo() const {
    if (!isTrained || model == nullptr) {
        std::cout << "Model is not trained" << std::endl;
        return;
    }
    
    std::cout << "\n=== SVM Model Information ===" << std::endl;
    std::cout << "Number of support vectors: " << model->l << std::endl;
    std::cout << "Number of classes: " << model->nr_class << std::endl;
    std::cout << "Number of features: " << model->SV[0] != nullptr ? "computed" : "unknown" << std::endl;
    std::cout << "Kernel type: RBF" << std::endl;
    std::cout << "C parameter: " << parameters.C << std::endl;
    std::cout << "Accuracy: " << (accuracy * 100.0) << "%" << std::endl;
    std::cout << "============================\n" << std::endl;
}