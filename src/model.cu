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
    use_scaling     = true; // Enable scaling by default
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

SVMmodel::SVMmodel(const SVMmodel &other)
    : is_trained(other.is_trained), n_features(other.n_features), svm_model(other.svm_model),
      C(other.C), kernel_type(other.kernel_type), kernel(other.kernel),
      gamma_type(other.gamma_type), gamma(other.gamma), tolerance(other.tolerance),
      cache_size(other.cache_size), max_iter(other.max_iter), nochange_steps(other.nochange_steps),
      verbosity(other.verbosity), n_support(other.n_support), bias(other.bias),
      n_classes(other.n_classes), feature_means(other.feature_means), 
      feature_stds(other.feature_stds), use_scaling(other.use_scaling) {}

SVMmodel::SVMmodel(SVMmodel &&other) noexcept
    : is_trained(other.is_trained), n_features(other.n_features), svm_model(other.svm_model),
      C(other.C), kernel_type(move(other.kernel_type)), kernel(other.kernel),
      gamma_type(move(other.gamma_type)), gamma(other.gamma), tolerance(other.tolerance),
      cache_size(other.cache_size), max_iter(other.max_iter), nochange_steps(other.nochange_steps),
      verbosity(other.verbosity), n_support(other.n_support), bias(other.bias),
      n_classes(other.n_classes), feature_means(move(other.feature_means)),
      feature_stds(move(other.feature_stds)), use_scaling(other.use_scaling) {
    other.svm_model = nullptr;
}

SVMmodel::~SVMmodel() {
    if (svm_model) {
        svm_free_and_destroy_model(&svm_model);
    }
}

SVMmodel &SVMmodel::operator=(const SVMmodel &other) {
    if (this != &other) {
        if (svm_model) {
            svm_free_and_destroy_model(&svm_model);
            svm_model = nullptr;
        }

        is_trained     = other.is_trained;
        n_features     = other.n_features;
        C              = other.C;
        kernel_type    = other.kernel_type;
        kernel         = other.kernel;
        gamma_type     = other.gamma_type;
        gamma          = other.gamma;
        tolerance      = other.tolerance;
        cache_size     = other.cache_size;
        max_iter       = other.max_iter;
        nochange_steps = other.nochange_steps;
        verbosity      = other.verbosity;
        n_support      = other.n_support;
        bias           = other.bias;
        n_classes      = other.n_classes;
        feature_means  = other.feature_means;
        feature_stds   = other.feature_stds;
        use_scaling    = other.use_scaling;

        svm_model = other.svm_model;
    }
    return *this;
}

SVMmodel &SVMmodel::operator=(SVMmodel &&other) noexcept {
    if (this != &other) {
        if (svm_model) {
            svm_free_and_destroy_model(&svm_model);
        }

        is_trained     = other.is_trained;
        n_features     = other.n_features;
        svm_model      = other.svm_model;
        C              = other.C;
        kernel_type    = move(other.kernel_type);
        kernel         = other.kernel;
        gamma_type     = move(other.gamma_type);
        gamma          = other.gamma;
        tolerance      = other.tolerance;
        cache_size     = other.cache_size;
        max_iter       = other.max_iter;
        nochange_steps = other.nochange_steps;
        verbosity      = other.verbosity;
        n_support      = other.n_support;
        bias           = other.bias;
        n_classes      = other.n_classes;
        feature_means  = move(other.feature_means);
        feature_stds   = move(other.feature_stds);
        use_scaling    = other.use_scaling;

        other.svm_model = nullptr;
    }
    return *this;
}

// === SCALING HELPERS ===
void SVMmodel::compute_statistics(const vector<vector<double>>& data) {
    if (data.empty()) return;
    
    int n_samples = data.size();
    int n_dims = data[0].size();
    
    feature_means.assign(n_dims, 0.0);
    feature_stds.assign(n_dims, 0.0);
    
    // Calculate Mean
    for (const auto& sample : data) {
        for (int j = 0; j < n_dims; j++) {
            feature_means[j] += sample[j];
        }
    }
    for (int j = 0; j < n_dims; j++) {
        feature_means[j] /= n_samples;
    }
    
    // Calculate Std Dev
    for (const auto& sample : data) {
        for (int j = 0; j < n_dims; j++) {
            feature_stds[j] += pow(sample[j] - feature_means[j], 2);
        }
    }
    for (int j = 0; j < n_dims; j++) {
        feature_stds[j] = sqrt(feature_stds[j] / n_samples);
        // Avoid division by zero for constant features
        if (feature_stds[j] < 1e-9) feature_stds[j] = 1.0;
    }
}

double SVMmodel::scale_value(double value, int feature_idx) const {
    if (!use_scaling || feature_idx >= feature_means.size()) return value;
    return (value - feature_means[feature_idx]) / feature_stds[feature_idx];
}
// =======================

float* SVMmodel::convertToDeviceArray(const vector<vector<double>>& data, int& n_rows, int& n_cols) {
    n_rows = data.size();
    n_cols = data[0].size();
    float* h_data = new float[n_rows * n_cols];
    for (int j = 0; j < n_cols; j++) {
        for (int i = 0; i < n_rows; i++) {
            h_data[j * n_rows + i] = static_cast<float>(data[i][j]);
        }
    }
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

static string libsvm_buffer = "";
static int libsvm_epoch_counter = 0;
static Timer* libsvm_timer = nullptr;
static bool capturing_raw = false;
static vector<string> raw_lines;
static string current_iter = "N/A";

static void print_libsvm_progress(const char *s) {
    libsvm_buffer += s;
    size_t newline_pos = libsvm_buffer.find('\n');
    while (newline_pos != string::npos) {
        string line = libsvm_buffer.substr(0, newline_pos);
        libsvm_buffer = libsvm_buffer.substr(newline_pos + 1);
        
        if (line.find("optimization finished") != string::npos) {
            capturing_raw = true;
            raw_lines.clear();
            raw_lines.push_back(line);
            size_t iter_pos = line.find("#iter = ");
            if (iter_pos != string::npos) {
                size_t iter_end = line.find(",", iter_pos);
                if (iter_end == string::npos) iter_end = line.length();
                current_iter = line.substr(iter_pos + 8, iter_end - (iter_pos + 8));
                while (!current_iter.empty() && isspace(current_iter.back())) current_iter.pop_back();
            }
        }
        else if (capturing_raw) {
            bool is_data_line = (line.find("nu = ") != string::npos || 
                                 line.find("obj = ") != string::npos || 
                                 line.find("nSV") != string::npos || 
                                 line.find("rho") != string::npos);
            if (is_data_line) {
                raw_lines.push_back(line);
            } else if (line.find("*") != string::npos || 
                       (line.empty() && raw_lines.size() > 1) ||
                       line.find("....") != string::npos) {
                capturing_raw = false;
                libsvm_epoch_counter++;
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
                float elapsed_time = 0.0f;
                if (libsvm_timer) {
                    libsvm_timer->stop();
                    elapsed_time = libsvm_timer->get();
                }
                printf("Classifier %d: #iter = %s, loss = %s, time = %s\n", 
                       libsvm_epoch_counter, current_iter.c_str(), obj_str.c_str(), format_time(elapsed_time).c_str());
                raw_lines.clear();
            }
        }
        newline_pos = libsvm_buffer.find('\n');
    }
}

void SVMmodel::train(const vector<vector<double>>& data, const vector<int>& labels) {
    libsvm_buffer.clear();
    svm_set_print_string_function(&print_libsvm_progress);
    
    puts("=======================TRAINING START=======================");
    cout << "Starting SVM training with LIBSVM..." << endl;
    
    Timer timer;
    timer.start();
    
    int n_samples = data.size();
    if (n_samples == 0) throw runtime_error("Training data is empty!");
    
    n_features = data[0].size();
    if (n_samples != static_cast<int>(labels.size())) throw runtime_error("Data and labels size mismatch!");
    
    cout << "Number of samples: " << n_samples << endl;
    cout << "Number of features: " << n_features << endl;

    // === SCALING STEP ===
    if (use_scaling) {
        cout << "Computing feature statistics and scaling data..." << endl;
        compute_statistics(data);
    }

    if (gamma_type == "auto") {
        gamma = 1.0f / n_features;
    } else {
        gamma = std::stof(gamma_type);
    }
    
    svm_problem prob;
    prob.l = n_samples;
    prob.y = new double[n_samples];
    prob.x = new svm_node*[n_samples];
    
    for (int i = 0; i < n_samples; i++) {
        prob.y[i] = static_cast<double>(labels[i]);
        
        prob.x[i] = new svm_node[n_features + 1];
        
        for (int j = 0; j < n_features; j++) {
            prob.x[i][j].index = j + 1;
            // Apply scaling here
            prob.x[i][j].value = use_scaling ? scale_value(data[i][j], j) : data[i][j];
        }
        prob.x[i][n_features].index = -1;
        prob.x[i][n_features].value = 0;
    }
    
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
    cout << "  Scaling: " << (use_scaling ? "Enabled (Z-score)" : "Disabled") << endl;

    set<int> unique_labels(labels.begin(), labels.end());
    int n_unique_classes = unique_labels.size();
    
    libsvm_epoch_counter = 0;
    libsvm_timer = &timer;
    
    const char* error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
        for (int i = 0; i < n_samples; i++) delete[] prob.x[i];
        delete[] prob.x;
        delete[] prob.y;
        throw runtime_error(string("SVM parameter error: ") + error_msg);
    }
    
    if (svm_model) {
        svm_free_and_destroy_model(&svm_model);
    }
    svm_model = svm_train(&prob, &param);
    
    timer.stop();
    float training_time = timer.get();
    libsvm_timer = nullptr;
    
    n_support = svm_model->l;
    n_classes = svm_model->nr_class;
    bias = -svm_model->rho[0];
    
    for (int i = 0; i < n_samples; i++) delete[] prob.x[i];
    delete[] prob.x;
    delete[] prob.y;
    
    is_trained = true;
    
    printf(" - Time = %s\n", format_time(training_time).c_str());
    puts("\n========================TRAINING END========================");
    cout << "Number of support vectors: " << n_support << endl;
    printf("Total training time: %s\n\n", format_time(training_time).c_str());
}

vector<int> SVMmodel::predict(const vector<vector<double>>& samples) const {
    if (!is_trained || !svm_model) throw runtime_error("Model must be trained before testing!");
    
    int n_samples = samples.size();
    if (n_samples == 0) throw runtime_error("Test data is empty!");
    if (static_cast<int>(samples[0].size()) != n_features) throw runtime_error("Dimension mismatch!");
    
    cout << "\n======================PREDICTION START======================" << endl;
    
    Timer timer;
    timer.start();
    
    vector<int> predictions(n_samples);
    
    // Optimization: Pre-allocate ONE node buffer
    svm_node* x = new svm_node[n_features + 1];
    x[n_features].index = -1;
    x[n_features].value = 0;

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            x[j].index = j + 1;
            // Apply scaling using stats stored during training
            x[j].value = use_scaling ? scale_value(samples[i][j], j) : samples[i][j];
        }
        
        predictions[i] = static_cast<int>(svm_predict(svm_model, x));
        
        bool should_display = false;
        if ((i + 1) % 1000 == 0) should_display = true;
        else if (n_samples >= 4 && ((i+1) % (n_samples/4) == 0)) should_display = true;
        
        if (should_display) {
            float elapsed_time = timer.get();
            float progress = ((i + 1) * 100.0f) / n_samples;
            printf("Predicted %d/%d (%.1f%%), time = %s\n", 
                   i + 1, n_samples, progress, format_time(elapsed_time).c_str());
            fflush(stdout);
        }
    }
    
    delete[] x; // Clean up the single buffer
    
    timer.stop();
    float total_time = timer.get();
    
    puts("");
    cout << "=======================PREDICTION END=======================" << endl;
    printf("Total prediction time: %s\n", format_time(total_time).c_str());
    return predictions;
}

double SVMmodel::calculateAccuracy(const vector<int>& predicted, const vector<int>& actual, int numClasses) {
    if (predicted.size() != actual.size() || predicted.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < actual.size(); i++) {
        if (predicted[i] == actual[i]) correct++;
    }
    return static_cast<double>(correct) / actual.size();
}

vector<vector<int>> SVMmodel::calculateClassificationReport(const vector<int>& predicted, const vector<int>& actual, int numClasses) {
    vector<vector<int>> report(numClasses, vector<int>(3, 0));
    for (size_t i = 0; i < actual.size(); i++) {
        if (predicted[i] == actual[i]) report[actual[i]][0]++;
        else {
            report[predicted[i]][1]++;
            report[actual[i]][2]++;
        }
    }
    return report;
}

vector<vector<int>> SVMmodel::calculateConfusionMatrix(const vector<int>& predicted, const vector<int>& actual, int numClasses) {
    vector<vector<int>> confusionMatrix(numClasses, vector<int>(numClasses, 0));
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] >= 0 && actual[i] < numClasses && predicted[i] >= 0 && predicted[i] < numClasses) {
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
        cout << i << "\t" << precision << "\t\t" << recall << "\t" << f1_score << "\t\t" << support << endl;
    }
}

void SVMmodel::printConfusionMatrix(const vector<vector<int>>& confusionMatrix) {
    cout << "\nConfusion Matrix:" << endl;
    cout << "Actual \\ Predicted | ";
    int numClasses = confusionMatrix.size();
    for (int i = 0; i < numClasses; i++) cout << i << " ";
    cout << endl << "--------------------" << endl;
    for (int i = 0; i < numClasses; i++) {
        cout << i << " | ";
        for (int j = 0; j < numClasses; j++) cout << confusionMatrix[i][j] << " ";
        cout << endl;
    }
}

bool SVMmodel::save(const string& modelPath) const {
    if (!is_trained || !svm_model) return false;
    
    if (svm_save_model(modelPath.c_str(), svm_model) == 0) {
        string metaPath = modelPath + ".meta";
        ofstream metaFile(metaPath, ios::binary);
        if (metaFile.is_open()) {
            metaFile.write(reinterpret_cast<const char*>(&n_features), sizeof(int));
            // Save Scaling Params
            size_t size = feature_means.size();
            metaFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
            metaFile.write(reinterpret_cast<const char*>(feature_means.data()), size * sizeof(double));
            metaFile.write(reinterpret_cast<const char*>(feature_stds.data()), size * sizeof(double));
            metaFile.close();
        }
        cout << "Model saved successfully to " << modelPath << endl;
        return true;
    }
    return false;
}

bool SVMmodel::load(const string& modelPath) {
    if (svm_model) svm_free_and_destroy_model(&svm_model);
    svm_model = svm_load_model(modelPath.c_str());

    if (!svm_model) {
        cerr << "Error: Could not load model from " << modelPath << endl;
        return false;
    }
    n_support = svm_model->l;
    n_classes = svm_model->nr_class;

    string metaPath = modelPath + ".meta";
    ifstream metaFile(metaPath, ios::binary);
    if (metaFile.is_open()) {
        metaFile.read(reinterpret_cast<char*>(&n_features), sizeof(int));
        
        // Load Scaling Params
        size_t size;
        if (metaFile.read(reinterpret_cast<char*>(&size), sizeof(size_t))) {
            feature_means.resize(size);
            feature_stds.resize(size);
            metaFile.read(reinterpret_cast<char*>(feature_means.data()), size * sizeof(double));
            metaFile.read(reinterpret_cast<char*>(feature_stds.data()), size * sizeof(double));
            use_scaling = true;
            cout << "Loaded scaling stats for " << size << " features." << endl;
        } else {
            use_scaling = false;
            cout << "Warning: Old meta file format (no scaling stats). Scaling disabled." << endl;
        }
        metaFile.close();
    } else {
        // Fallback logic for n_features if meta missing (same as before)
        n_features = 0;
        if (n_support > 0 && svm_model->SV && svm_model->SV[0]) {
             // ... [Rest of your fallback logic] ...
             int samples_to_check = min(n_support, 100);
             for (int sv = 0; sv < samples_to_check; ++sv) {
                const svm_node* node = svm_model->SV[sv];
                if (!node) continue;
                for (int i = 0; node[i].index != -1; ++i) 
                    if (node[i].index > n_features) n_features = node[i].index;
             }
        }
        use_scaling = false; 
    }

    if (svm_model->rho) bias = -svm_model->rho[0]; else bias = 0.0f;
    
    switch (svm_model->param.kernel_type) {
        case LINEAR:   kernel_type = "LINEAR"; break;
        case RBF:      kernel_type = "RBF";    break;
        default:       kernel_type = "UNKNOWN"; break;
    }
    C = static_cast<float>(svm_model->param.C);
    gamma = static_cast<float>(svm_model->param.gamma);
    gamma_type = "value";

    is_trained = true;
    cout << "Model loaded: " << n_support << " SVs, " << n_classes << " classes." << endl;
    return true;
}

bool SVMmodel::getIsTrained() const { return is_trained; }

void SVMmodel::printModelInfo() const {
    if (!is_trained) { cout << "Model is not trained" << endl; return; }
    cout << "\n=== cuML SVM Model Information ===" << endl;
    cout << "Number of support vectors: " << n_support << endl;
    cout << "Number of classes: " << n_classes << endl;
    cout << "Kernel type: " << kernel_type << endl;
    cout << "C parameter: " << C << endl;
    cout << "Gamma: " << gamma << endl;
    cout << "Scaling: " << (use_scaling ? "Enabled" : "Disabled") << endl;
    cout << "============================\n" << endl;
}

bool SVMmodel::save_evaluation(double accuracy, const vector<vector<int>>& class_report, 
                              const vector<vector<int>>& conf_matrix, const string& eval_path) const {
    size_t found = eval_path.find_last_of("/\\");
    if (found != string::npos) {
        string dir_path = eval_path.substr(0, found);
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            #ifdef _WIN32
                _mkdir(dir_path.c_str());
            #else 
                mkdir(dir_path.c_str(), 0777);
            #endif
        }
    }
    ofstream ofs(eval_path);
    if (ofs.is_open()) {
        ofs << "SVM Test Accuracy: " << accuracy * 100.0 << "%\n";
        ofs << "Classification Report:\n";
        for (const auto& row : class_report) {
            for (const auto& val : row) ofs << val << " ";
            ofs << "\n";
        }
        ofs << "Confusion Matrix:\n";
        for (const auto& row : conf_matrix) {
            for (const auto& val : row) ofs << val << " ";
            ofs << "\n";
        }
        ofs.close();
        return true;
    }
    return false;
}