#include "model.h"

SVMmodel::SVMmodel() : is_trained(false), n_features(0),
    C(10.0f), kernel_type("RBF"), gamma_type("auto"),
    tolerance(1e-3f), cache_size(200.0f), max_iter(100), nochange_steps(100), verbosity(0) {
    
    kernel = (kernel_type == "RBF") ? ML::SVM::KernelType::RBF : ML::SVM::KernelType::LINEAR;
    gamma =  (gamma_type == "auto") ? 1.0 / n_features : std::stof(gamma_type);

    // Initialize model pointers
    n_support = 0;
    b = 0.0f;
    dual_coefs = nullptr;
    x_support = nullptr;
    support_idx = nullptr;
    n_classes = 0;
    unique_labels = nullptr;
}

SVMmodel::SVMmodel(float C, string kernel_type, string gamma_type) : SVMmodel() {
    this->C = C;
    this->kernel_type = kernel_type;
    this->kernel = (kernel_type == "RBF") ? ML::SVM::KernelType::RBF : ML::SVM::KernelType::LINEAR;
    this->gamma_type = gamma_type;
    this->gamma = (gamma_type == "auto") ? 1.0f / n_features : std::stof(gamma_type);
}

SVMmodel::SVMmodel(float C, string kernel_type, string gamma_type, float tolerance, float cache_size, int max_iter, int nochange_steps) : SVMmodel(C, kernel_type, gamma_type) {
    this->tolerance = tolerance;
    this->cache_size = cache_size;
    this->max_iter = max_iter;
    this->nochange_steps = nochange_steps;
}

SVMmodel::~SVMmodel() {
    if (is_trained) {
        // Free device memory
        if (dual_coefs) cudaFree(dual_coefs);
        if (x_support) cudaFree(x_support);
        if (support_idx) cudaFree(support_idx);
        if (unique_labels) cudaFree(unique_labels);
    }
    cumlDestroy(handle);
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

void SVMmodel::train(const vector<vector<double>>& data, const vector<int>& labels) {
    cout << "Starting SVM training..." << endl;
    
    int n_samples = data.size();
    if (n_samples == 0) {
        throw runtime_error("Training data is empty!");
    }
    
    n_features = data[0].size();
    if (n_samples != static_cast<int>(labels.size())) {
        throw runtime_error("Data and labels size mismatch!");
    }
    
    cout << "Number of samples: " << n_samples << endl;
    cout << "Number of features: " << n_features << endl;
    
    // Flatten 2D vector and convert double to float
    vector<float> flattened_data(n_samples * n_features);
    for (int i = 0; i < n_samples; i++) {
        if (static_cast<int>(data[i].size()) != n_features) {
            throw runtime_error("Inconsistent feature dimensions in training data!");
        }
        for (int j = 0; j < n_features; j++) {
            flattened_data[i * n_features + j] = static_cast<float>(data[i][j]);
        }
    }
    
    // Allocate device memory for training data
    rmm::device_uvector<float> d_train_data(n_samples * n_features, handle.get_stream());
    rmm::device_uvector<float> d_train_labels(n_samples, handle.get_stream());
    
    // Copy flattened data to device
    cudaMemcpy(d_train_data.data(), flattened_data.data(), 
                n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    
    // Convert int labels to float for cuML
    vector<float> float_labels(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float_labels[i] = static_cast<float>(labels[i]);
    }
    cudaMemcpy(d_train_labels.data(), float_labels.data(), 
                n_samples * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set SVM parameters
    ML::SVM::SvmParameter params;
    params.C = C_PARAM;                                                                                 // C parameter
    params.kernel = (KERNEL_PARAM == "rbf") ? ML::SVM::KernelType::RBF : ML::SVM::KernelType::LINEAR;   // Kernel type
    params.gamma = (GAMMA_TYPE == "auto") ? (1.0 / n_features) : GAMMA_TYPE;                            // gamma parameter

    params.tol = TOLERANCE;                 // Tolerance
    params.cache_size = CACHE_SIZE;         // Cache size in MB
    params.max_iter = MAX_ITER;             // No limit on iterations
    params.nochange_steps = NOCHANGE_STEPS; // Early stopping
    params.verbosity = VERBOSITY;           // Verbosity level
    
    cout << "Training with parameters:" << endl;
    cout << "  C: " << params.C << endl;
    cout << "  Kernel: " << params.kernel << " (" << KERNEL_PARAM << ")" << endl;
    cout << "  Gamma: " << params.gamma << " (" << GAMMA_TYPE << ")" << endl;
    
    // Train the SVM model
    svm_model.fit(handle, d_train_data.data(), n_samples, n_features,
                    d_train_labels.data(), params);
    
    is_trained = true;
    cout << "Training completed successfully!" << endl;
}

vector<int> SVMmodel::predict(const vector<vector<double>>& samples) const {
   if (!is_trained) {
        throw runtime_error("Model must be trained before testing!");
    }
    
    int n_samples = samples.size();
    if (n_samples == 0) {
        throw runtime_error("Test data is empty!");
    }
    
    if (static_cast<int>(samples[0].size()) != n_features) {
        throw runtime_error("Test data feature dimensions don't match training data!");
    }
    
    cout << "Predicting on " << n_samples << " test samples..." << endl;
    
    // Flatten 2D vector and convert double to float
    vector<float> flattened_data(n_samples * n_features);
    for (int i = 0; i < n_samples; i++) {
        if (static_cast<int>(samples[i].size()) != n_features) {
            throw runtime_error("Inconsistent feature dimensions in test data!");
        }
        for (int j = 0; j < n_features; j++) {
            flattened_data[i * n_features + j] = static_cast<float>(samples[i][j]);
        }
    }
    
    // Allocate device memory for test data
    rmm::device_uvector<float> d_test_data(n_samples * n_features, handle.get_stream());
    rmm::device_uvector<float> d_predictions(n_samples, handle.get_stream());
    
    // Copy flattened test data to device
    cudaMemcpy(d_test_data.data(), flattened_data.data(), 
                n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    
    // Predict
    svm_model.predict(handle, d_test_data.data(), n_samples, n_features,
                        d_predictions.data());
    
    // Copy predictions back to host
    vector<float> float_predictions(n_samples);
    cudaMemcpy(float_predictions.data(), d_predictions.data(), 
                n_samples * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Convert float predictions to int
    vector<int> predictions(n_samples);
    for (int i = 0; i < n_samples; i++) {
        predictions[i] = static_cast<int>(round(float_predictions[i]));
    }
    return predictions;
}

double SVMmodel::calculateAccuracy(const vector<int>& predicted, const vector<int>& actual, int numClasses) {
    if (predicted.size() != actual.size() || predicted.empty()) {
        cerr << "Error: Size mismatch or empty vectors for accuracy calculation" << endl;
        return 0.0;
    }
    
    int correct = 0;
    for (size_t i = 0; i < actual.size(); i++) {
        if (predicted[i] == actual[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / actual.size();
}

vector<vector<int>> SVMmodel::calculateConfusionMatrix(const vector<int>& predicted, const vector<int>& actual, int numClasses) {
    vector<vector<int>> confusionMatrix(numClasses, vector<int>(numClasses, 0));
    
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] >= 0 && actual[i] < numClasses && 
            predicted[i] >= 0 && predicted[i] < numClasses) {
            confusionMatrix[actual[i]][predicted[i]]++;
        }
    }
    return confusionMatrix;
}

void SVMmodel::printClassificationReport(const vector<vector<int>>& classificationReport) {
    cout << "\nClassification Report:" << endl;
    cout << "Class\tPrecision\tRecall\tF1-Score\tSupport" << endl;
    
    for (size_t i = 0; i < classificationReport.size(); i++) {
        int tp = classificationReport[i][0];
        int fp = classificationReport[i][1];
        int fn = classificationReport[i][2];
        int support = tp + fn;
        
        double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;
        
        cout << i << "\t" 
             << precision << "\t\t" 
             << recall << "\t" 
             << f1_score << "\t\t" 
             << support << endl;
    }
}

void SVMmodel::printConfusionMatrix(const vector<vector<int>>& confusionMatrix) {
    cout << "\nConfusion Matrix:" << endl;
    cout << "Actual \\ Predicted | ";
    int numClasses = confusionMatrix.size();

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
    if (!is_trained) {
        cerr << "Error: Model not trained or invalid" << endl;
        return false;
    }
    
    ofstream file(modelPath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open file for writing: " << modelPath << endl;
        return false;
    }
    
    // Write scalar parameters
    file.write(reinterpret_cast<const char*>(&n_features), sizeof(int));
    file.write(reinterpret_cast<const char*>(&n_support), sizeof(int));
    file.write(reinterpret_cast<const char*>(&n_classes), sizeof(int));
    file.write(reinterpret_cast<const char*>(&b), sizeof(float));
    file.write(reinterpret_cast<const char*>(&C), sizeof(float));
    file.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    file.write(reinterpret_cast<const char*>(&tolerance), sizeof(float));
    file.write(reinterpret_cast<const char*>(&cache_size), sizeof(float));
    file.write(reinterpret_cast<const char*>(&max_iter), sizeof(int));
    file.write(reinterpret_cast<const char*>(&nochange_steps), sizeof(int));
    file.write(reinterpret_cast<const char*>(&kernel_type), sizeof(cumlSvmKernelType));
    
    // Copy device arrays to host and write
    if (n_support > 0) {
        // dual_coefs: n_support elements
        float* h_dual_coefs = new float[n_support];
        cudaMemcpy(h_dual_coefs, dual_coefs, n_support * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(h_dual_coefs), n_support * sizeof(float));
        delete[] h_dual_coefs;
        
        // x_support: n_support * n_features elements
        float* h_x_support = new float[n_support * n_features];
        cudaMemcpy(h_x_support, x_support, n_support * n_features * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(h_x_support), n_support * n_features * sizeof(float));
        delete[] h_x_support;
        
        // support_idx: n_support elements
        int* h_support_idx = new int[n_support];
        cudaMemcpy(h_support_idx, support_idx, n_support * sizeof(int), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(h_support_idx), n_support * sizeof(int));
        delete[] h_support_idx;
    }
    
    if (n_classes > 0) {
        // unique_labels: n_classes elements
        float* h_unique_labels = new float[n_classes];
        cudaMemcpy(h_unique_labels, unique_labels, n_classes * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(h_unique_labels), n_classes * sizeof(float));
        delete[] h_unique_labels;
    }
    
    file.close();
    cout << "Model saved successfully to " << modelPath << endl;
    return true;
}

bool SVMmodel::load(const string& modelPath) {
    ifstream file(modelPath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open file for reading: " << modelPath << endl;
        return false;
    }
    
    // Free existing device memory if model was previously trained
    if (is_trained) {
        if (dual_coefs) cudaFree(dual_coefs);
        if (x_support) cudaFree(x_support);
        if (support_idx) cudaFree(support_idx);
        if (unique_labels) cudaFree(unique_labels);
    }
    
    // Read scalar parameters
    file.read(reinterpret_cast<char*>(&n_features), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_support), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_classes), sizeof(int));
    file.read(reinterpret_cast<char*>(&b), sizeof(float));
    file.read(reinterpret_cast<char*>(&C), sizeof(float));
    file.read(reinterpret_cast<char*>(&gamma), sizeof(float));
    file.read(reinterpret_cast<char*>(&tolerance), sizeof(float));
    file.read(reinterpret_cast<char*>(&cache_size), sizeof(float));
    file.read(reinterpret_cast<char*>(&max_iter), sizeof(int));
    file.read(reinterpret_cast<char*>(&nochange_steps), sizeof(int));
    file.read(reinterpret_cast<char*>(&kernel_type), sizeof(cumlSvmKernelType));
    
    // Read device arrays
    if (n_support > 0) {
        // dual_coefs
        float* h_dual_coefs = new float[n_support];
        file.read(reinterpret_cast<char*>(h_dual_coefs), n_support * sizeof(float));
        cudaMalloc(&dual_coefs, n_support * sizeof(float));
        cudaMemcpy(dual_coefs, h_dual_coefs, n_support * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_dual_coefs;
        
        // x_support
        float* h_x_support = new float[n_support * n_features];
        file.read(reinterpret_cast<char*>(h_x_support), n_support * n_features * sizeof(float));
        cudaMalloc(&x_support, n_support * n_features * sizeof(float));
        cudaMemcpy(x_support, h_x_support, n_support * n_features * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_x_support;
        
        // support_idx
        int* h_support_idx = new int[n_support];
        file.read(reinterpret_cast<char*>(h_support_idx), n_support * sizeof(int));
        cudaMalloc(&support_idx, n_support * sizeof(int));
        cudaMemcpy(support_idx, h_support_idx, n_support * sizeof(int), cudaMemcpyHostToDevice);
        delete[] h_support_idx;
    }
    
    if (n_classes > 0) {
        // unique_labels
        float* h_unique_labels = new float[n_classes];
        file.read(reinterpret_cast<char*>(h_unique_labels), n_classes * sizeof(float));
        cudaMalloc(&unique_labels, n_classes * sizeof(float));
        cudaMemcpy(unique_labels, h_unique_labels, n_classes * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_unique_labels;
    }
    
    file.close();
    is_trained = true;
    cout << "Model loaded successfully from " << modelPath << endl;
    return true;
}

bool SVMmodel::getIsTrained() const {
    return is_trained;
}

void SVMmodel::printModelInfo() const {
    if (!is_trained) {
        cout << "Model is not trained" << endl;
        return;
    }
    
    cout << "\n=== cuML SVM Model Information ===" << endl;
    cout << "Number of support vectors: " << n_support << endl;
    cout << "Number of classes: " << n_classes << endl;
    cout << "Number of features: " << n_features << endl;
    cout << "Kernel type: " << kernel_type << endl;
    cout << "C parameter: " << C << endl;
    cout << "Gamma: " << gamma << endl;
    cout << "============================\n" << endl;
}