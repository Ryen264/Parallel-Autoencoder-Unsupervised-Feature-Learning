#include "model.h"

SVMmodel::SVMmodel() : C(10.0f), kernel_type("RBF"), gamma_type("auto"),
    tolerance(1e-3f), cache_size(200.0f), max_iter(100), nochange_steps(100), verbosity(0) {
    kernel = (kernel_type == "RBF") ? RBF : LINEAR;
    gamma  = 0.0f;

    svm_model       = nullptr;
    is_trained      = false;
    n_features      = 0;

    n_support       = 0;
    bias            = 0.0f;
    n_classes       = 0;
}

SVMmodel::SVMmodel(float C, string kernel_type, string gamma_type) : SVMmodel() {
    this->C             = C;
    this->kernel_type   = kernel_type;
    this->kernel        = (kernel_type == "RBF") ? RBF : LINEAR;
    this->gamma_type    = gamma_type;
}

SVMmodel::SVMmodel(float C, string kernel_type, string gamma_type,
    float tolerance, float cache_size, int max_iter, int nochange_steps) : SVMmodel(C, kernel_type, gamma_type) {
    this->tolerance    = tolerance;
    this->cache_size   = cache_size;
    this->max_iter     = max_iter;
    this->nochange_steps = nochange_steps;
}

SVMmodel::~SVMmodel() {
    if (svm_model) {
        svm_free_and_destroy_model(&svm_model);
    }
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

// Custom LIBSVM output handler for progress tracking
static string libsvm_buffer = "";
static int libsvm_epoch_counter = 0;
static Timer* libsvm_timer = nullptr;
static bool capturing_raw = false;
static vector<string> raw_lines;
static string current_iter = "N/A";

static void print_libsvm_progress(const char *s) {
    // Append all incoming text
    libsvm_buffer += s;
    
    // Process complete lines
    size_t newline_pos = libsvm_buffer.find('\n');
    while (newline_pos != string::npos) {
        string line = libsvm_buffer.substr(0, newline_pos);
        libsvm_buffer = libsvm_buffer.substr(newline_pos + 1);
        
        // Start capturing when optimization finished appears
        if (line.find("optimization finished") != string::npos) {
            capturing_raw = true;
            raw_lines.clear();
            raw_lines.push_back(line);
            
            // Extract #iter from this line
            size_t iter_pos = line.find("#iter = ");
            if (iter_pos != string::npos) {
                size_t iter_end = line.find(",", iter_pos);
                if (iter_end == string::npos) iter_end = line.length();
                current_iter = line.substr(iter_pos + 8, iter_end - (iter_pos + 8));
                while (!current_iter.empty() && isspace(current_iter.back())) current_iter.pop_back();
            }
        }
        // Continue capturing subsequent lines (nu, obj, nSV, etc.)
        else if (capturing_raw) {
            // Check if this line contains useful information
            bool is_data_line = (line.find("nu = ") != string::npos || 
                                 line.find("obj = ") != string::npos || 
                                 line.find("nSV") != string::npos || 
                                 line.find("rho") != string::npos);
            
            if (is_data_line) {
                // Continue capturing data lines
                raw_lines.push_back(line);
            } else if (line.find("*") != string::npos || 
                       (line.empty() && raw_lines.size() > 1) ||
                       line.find("....") != string::npos) {
                // Stop capturing when we see separator or empty line after data
                capturing_raw = false;
                libsvm_epoch_counter++;
                
                // Parse obj from captured lines
                string obj_str = "N/A";
                for (const auto& raw_line : raw_lines) {
                    size_t obj_pos = raw_line.find("obj = ");
                    if (obj_pos != string::npos) {
                        size_t obj_end = raw_line.find(",", obj_pos);
                        if (obj_end == string::npos) obj_end = raw_line.length();
                        obj_str = raw_line.substr(obj_pos + 6, obj_end - (obj_pos + 6));
                        while (!obj_str.empty() && isspace(obj_str.back())) obj_str.pop_back();
                        break;
                    }
                }
                
                // Get elapsed time
                float elapsed_time = 0.0f;
                if (libsvm_timer) {
                    libsvm_timer->stop();
                    elapsed_time = libsvm_timer->get();
                }
                
                // Display formatted output
                printf("Classifier %d: #iter = %s, loss = %s, time = %s\n", 
                       libsvm_epoch_counter, 
                       current_iter.c_str(), 
                       obj_str.c_str(),
                       format_time(elapsed_time).c_str());
                
                // Print all raw lines
                printf("  [Raw Output:\n");
                for (const auto& raw_line : raw_lines) {
                    printf("    %s\n", raw_line.c_str());
                }
                printf("  ]\n");
                fflush(stdout);
                
                raw_lines.clear();
            }
            // If not a data line and not a stop condition, just skip it
        }
        
        newline_pos = libsvm_buffer.find('\n');
    }
}

void SVMmodel::train(const vector<vector<double>>& data, const vector<int>& labels) {
    // Use custom LIBSVM output handler to display progress
    libsvm_buffer.clear();
    svm_set_print_string_function(&print_libsvm_progress);
    
    puts("=======================TRAINING START=======================");
    cout << "Starting SVM training with LIBSVM..." << endl;
    
    Timer timer;
    timer.start();
    
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
    
    // Set gamma if auto
    if (gamma_type == "auto") {
        gamma = 1.0f / n_features;
    } else {
        gamma = std::stof(gamma_type);
    }
    
    // Prepare svm_problem structure
    svm_problem prob;
    prob.l = n_samples;
    prob.y = new double[n_samples];
    prob.x = new svm_node*[n_samples];
    
    // Convert data to libsvm format
    for (int i = 0; i < n_samples; i++) {
        prob.y[i] = static_cast<double>(labels[i]);
        
        // Allocate nodes for this sample (n_features + 1 for terminating node)
        prob.x[i] = new svm_node[n_features + 1];
        
        for (int j = 0; j < n_features; j++) {
            prob.x[i][j].index = j + 1;  // libsvm uses 1-based indexing
            prob.x[i][j].value = data[i][j];
        }
        // Terminating node
        prob.x[i][n_features].index = -1;
        prob.x[i][n_features].value = 0;
    }
    
    // Set SVM parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = (kernel_type == "RBF") ? RBF : LINEAR;
    param.degree = 3;
    param.gamma = gamma;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = cache_size;
    param.C = C;
    param.eps = tolerance;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
    
    cout << "\nTraining with parameters:" << endl;
    cout << "  C: " << param.C << endl;
    cout << "  Kernel: " << kernel_type << endl;
    cout << "  Gamma: " << param.gamma << endl;
    cout << "  Max iterations: " << max_iter << endl;
    cout << "  Tolerance: " << tolerance << endl;
    cout << "  Cache size: " << cache_size << " MB" << endl;
    cout << "  Shrinking: " << (param.shrinking ? "enabled" : "disabled") << endl;
    
    // Count unique classes
    set<int> unique_labels(labels.begin(), labels.end());
    int n_unique_classes = unique_labels.size();
    int n_binary_classifiers = (n_unique_classes * (n_unique_classes - 1)) / 2;
    
    cout << "\nTraining information:" << endl;
    cout << "  Training samples: " << n_samples << endl;
    cout << "  Features per sample: " << n_features << endl;
    cout << "  Number of classes: " << n_unique_classes << endl;
    cout << "  Binary classifiers (one-vs-one): " << n_binary_classifiers << endl;
    cout << "  Batch size: " << n_samples << " (full dataset)" << endl;
    cout << endl;
    
    cout << "Training Progress (optimizing " << n_binary_classifiers << " binary classifiers):" << endl;
    
    // Initialize epoch counter and timer for LIBSVM progress tracking
    libsvm_epoch_counter = 0;
    libsvm_timer = &timer;
    
    // Check parameters
    const char* error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
        // Clean up
        for (int i = 0; i < n_samples; i++) {
            delete[] prob.x[i];
        }
        delete[] prob.x;
        delete[] prob.y;
        throw runtime_error(string("SVM parameter error: ") + error_msg);
    }
    
    // Train the model
    if (svm_model) {
        svm_free_and_destroy_model(&svm_model);
    }
    svm_model = svm_train(&prob, &param);
    
    timer.stop();
    float training_time = timer.get();
    libsvm_timer = nullptr;  // Clear timer reference
    
    // Store model information
    n_support = svm_model->l;
    n_classes = svm_model->nr_class;
    bias = -svm_model->rho[0];  // Bias term
    
    // Clean up problem data
    for (int i = 0; i < n_samples; i++) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;
    
    is_trained = true;
    
    printf(" - Time = %s\n", format_time(training_time).c_str());
    puts("\n========================TRAINING END========================");
    cout << "Training completed successfully!" << endl;
    cout << "Number of support vectors: " << n_support << endl;
    cout << "Number of classes: " << n_classes << endl;
    printf("Total training time: %s\n\n", format_time(training_time).c_str());
}

vector<int> SVMmodel::predict(const vector<vector<double>>& samples) const {
    if (!is_trained || !svm_model) {
        throw runtime_error("Model must be trained before testing!");
    }
    
    int n_samples = samples.size();
    if (n_samples == 0) {
        throw runtime_error("Test data is empty!");
    }
    
    if (static_cast<int>(samples[0].size()) != n_features) {
        throw runtime_error("Test data feature dimensions don't match training data!");
    }
    
    cout << "\n======================PREDICTION START======================" << endl;
    cout << "Predicting on " << n_samples << " test samples..." << endl;
    
    cout << "\nPrediction information:" << endl;
    cout << "  Test samples: " << n_samples << endl;
    cout << "  Features per sample: " << n_features << endl;
    cout << "  Number of classes: " << n_classes << endl;
    cout << "  Number of support vectors: " << n_support << endl;
    cout << endl;
    
    cout << "Prediction Progress:" << endl;
    
    Timer timer;
    timer.start();
    
    vector<int> predictions(n_samples);
    
    // Predict each sample
    for (int i = 0; i < n_samples; i++) {
        // Convert sample to libsvm format
        svm_node* x = new svm_node[n_features + 1];
        
        for (int j = 0; j < n_features; j++) {
            x[j].index = j + 1;  // libsvm uses 1-based indexing
            x[j].value = samples[i][j];
        }
        x[n_features].index = -1;  // Terminating node
        x[n_features].value = 0;
        
        // Predict
        double pred = svm_predict(svm_model, x);
        predictions[i] = static_cast<int>(pred);
        
        delete[] x;
        
        // Display progress at milestones
        bool should_display = false;
        
        if ((i + 1) % 1000 == 0) {
            should_display = true;
        } else if (n_samples >= 4) {
            // Show at 25%, 50%, 75%, 100%
            if ((i + 1) == n_samples / 4 || (i + 1) == n_samples / 2 || 
                (i + 1) == 3 * n_samples / 4 || (i + 1) == n_samples) {
                should_display = true;
            }
        } else if ((i + 1) == n_samples) {
            // Always show at end
            should_display = true;
        }
        
        if (should_display) {
            float elapsed_time = timer.get();
            float progress = ((i + 1) * 100.0f) / n_samples;
            printf("Predicted %d/%d samples (%.1f%%), time = %s\n", 
                   i + 1, n_samples, progress, format_time(elapsed_time).c_str());
            fflush(stdout);
        }
    }
    
    timer.stop();
    float total_time = timer.get();
    float avg_time_per_sample = total_time / n_samples;
    
    puts("");
    cout << "=======================PREDICTION END=======================" << endl;
    cout << "Prediction completed!" << endl;
    printf("Total prediction time: %s\n", format_time(total_time).c_str());
    printf("Average time per sample: %s\n\n", format_time(avg_time_per_sample).c_str());
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

vector<vector<int>> SVMmodel::calculateClassificationReport(const vector<int>& predicted, const vector<int>& actual, int numClasses) {
    vector<vector<int>> report(numClasses, vector<int>(3, 0)); // TP, FP, FN for each class
    
    for (size_t i = 0; i < actual.size(); i++) {
        if (predicted[i] == actual[i]) {
            report[actual[i]][0]++; // True Positive
        } else {
            report[predicted[i]][1]++; // False Positive
            report[actual[i]][2]++;    // False Negative
        }
    }
    return report;
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
    if (!is_trained || !svm_model) {
        cerr << "Error: Model not trained or invalid" << endl;
        return false;
    }
    
    // Use libsvm's save function
    if (svm_save_model(modelPath.c_str(), svm_model) == 0) {
        cout << "Model saved successfully to " << modelPath << endl;
        return true;
    } else {
        cerr << "Error: Could not save model to " << modelPath << endl;
        return false;
    }
}

bool SVMmodel::load(const string& modelPath) {
    // Free existing model if any
    if (svm_model) {
        svm_free_and_destroy_model(&svm_model);
    }
    
    // Use libsvm's load function
    svm_model = svm_load_model(modelPath.c_str());
    
    if (!svm_model) {
        cerr << "Error: Could not load model from " << modelPath << endl;
        return false;
    }
    
    // Extract model information
    n_support = svm_model->l;
    n_classes = svm_model->nr_class;
    n_features = svm_model->SV[0][0].index;  // Get number of features from first support vector
    
    // Count actual number of features
    for (int i = 0; svm_model->SV[0][i].index != -1; i++) {
        if (svm_model->SV[0][i].index > n_features) {
            n_features = svm_model->SV[0][i].index;
        }
    }
    
    bias = -svm_model->rho[0];
    
    is_trained = true;
    cout << "Model loaded successfully from " << modelPath << endl;
    cout << "Number of support vectors: " << n_support << endl;
    cout << "Number of classes: " << n_classes << endl;
    cout << "Number of features: " << n_features << endl;
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