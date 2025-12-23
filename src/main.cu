#include "constants.h"
#include "data_loader.h"
#include "cpu_data_loader.h"
#include "cpu_autoencoder.h"
#include "gpu_autoencoder.h"
#include "model.h"
#include "visualization.h"

#include <iostream>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
using namespace std;

string RUN_MODE       = "phase_2";  // "phase_1", "phase_2", "all"
string HARDWARE_MODE  = "cpu";  // "cpu", "gpu"

// NUM_BATCHES      = 1;  // CIFAR-10 has 5 training batches
// N_EPOCH          = 20;
// BATCH_SIZE       = 32;
// LEARNING_RATE    = 0.001f;
// VERBOSE          = false;
// CHECKPOINT       = 0;

const char *DATASET_DIR             = "./data/cifar-10-batches-bin";
const char *AUTOENCODER_OUTPUT_DIR  = "./output";
const char *CPU_AUTOENCODER_PATH    = "./cpu_autoencoder_model.bin";
const char *GPU_AUTOENCODER_PATH    = "./gpu_autoencoder_model.bin";
const char *ENCODED_DATASET_PATH    = "./encoded_dataset.bin";

const char *LABELS_FILE             = "labels.bin";
const char *MODEL_OUTPUT_DIR        = "./model";
const char *SVM_MODEL_FILE          = "svm_model.bin";

const char *VISUALIZATION_TRAINING_TIMES_SVG = "training_times.svg";
const char *VISUALIZATION_TRAINING_TIMES_CSV = "training_times.csv";
const char *VISUALIZATION_SPEEDUP_GRAPH_SVG  = "speedup_graph.svg";
const char *VISUALIZATION_SPEEDUP_GRAPH_CSV  = "speedup_data.csv";
const char *VISUALIZATION_HTML_DASHBOARD     = "performance_analysis.html";

// Load and preprocess dataset
Dataset load_dataset(const char *dataset_dir, int n_batches = NUM_BATCHES, bool is_train = true) {
    // Read dataset
    Dataset dataset = read_dataset(dataset_dir, n_batches, is_train);

    // Shuffle dataset
    shuffle_dataset(dataset);
    return dataset;
}

// Load labels from file
vector<int> load_labels(const char *labels_path, int n_samples = NUM_TRAIN_SAMPLES) {
    vector<int> labels(n_samples);
    FILE *lf = fopen(labels_path, "rb");
    if (lf) {
        size_t labels_read = fread(labels.data(), sizeof(int), n_samples, lf);
        if (labels_read != n_samples) {
            cerr << "Warning: Expected " << n_samples << " labels, but read " << labels_read << endl;
        }
        fclose(lf);
    } else {
        cerr << "Warning: Labels file not found, using random labels" << endl;
        for (int i = 0; i < n_samples; ++i) {
            labels[i] = rand() % NUM_CLASSES;
        }
    }
    return labels;
}

// Create a dummy dataset with random values for phase 2 testing
Dataset dummy_dataset(int n = NUM_TEST_SAMPLES, int width = IMAGE_WIDTH, int height = IMAGE_HEIGHT, int depth = IMAGE_DEPTH) {
    unique_ptr<float[]> data(new float[n * width * height * depth]);
    for (int i = 0; i < n * width * height * depth; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Dataset(data, n, width, height, depth);
}

// Create dummy labels for phase 2 testing
vector<int> dummy_labels(int n = NUM_TEST_SAMPLES, int num_classes = NUM_CLASSES) {
    vector<int> labels(n);
    for (int i = 0; i < n; ++i) {
        labels[i] = rand() % num_classes;
    }
    return labels;
}

// Phase 1: Train and encode by CPU autoencoder
Dataset phase_1_cpu(const Dataset& dataset, const char *output_dir, const char *autoencoder_path, const char *encoded_dataset_path,
                    int n_epoch = N_EPOCH, int batch_size = BATCH_SIZE, float learning_rate = LEARNING_RATE, bool verbose = VERBOSE, int checkpoint = CHECKPOINT,
                    bool is_save_model = true, bool is_save_encoded = true) {
    // Create and train model
    Cpu_Autoencoder autoencoder;
    printf("Training CPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    // Eval
    printf("CPU Autoencoder MSE = %.4f", autoencoder.eval(dataset));

    // Save model
    if (is_save_model)
        autoencoder.save_parameters(autoencoder_path);

    // Save encoded dataset
    Dataset encoded_dataset = autoencoder.encode(dataset);
    if (is_save_encoded)
        write_data(encoded_dataset, encoded_dataset_path);

    return encoded_dataset;
}

// Phase 1: Train and encode by GPU autoencoder
Dataset phase_1_gpu(const Dataset& dataset, const char *output_dir, const char *autoencoder_path, const char *encoded_dataset_path,
                    int n_epoch = N_EPOCH, int batch_size = BATCH_SIZE, float learning_rate = LEARNING_RATE, bool verbose = VERBOSE, int checkpoint = CHECKPOINT,
                    bool is_save_model = true, bool is_save_encoded = true) {
    // Create and train model
    Gpu_Autoencoder autoencoder;
    printf("Training GPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    // Eval
    printf("GPU Autoencoder MSE = %.4f", autoencoder.eval(dataset));

    // Save model
    if (is_save_model)
        autoencoder.save_parameters(autoencoder_path);

    // Save encoded dataset
    Dataset encoded_dataset = autoencoder.encode(dataset);
    if (is_save_encoded)
        write_data(encoded_dataset, encoded_dataset_path);

    return encoded_dataset;
}

template <typename AE>
void phase_1_train(const Dataset& dataset, AE& autoencoder, const char *output_dir, const char *autoencoder_path,
                    int n_epoch = N_EPOCH, int batch_size = BATCH_SIZE, float learning_rate = LEARNING_RATE, bool verbose = VERBOSE, int checkpoint = CHECKPOINT,
                    bool is_save_model = true) {
    // Create and train model
    printf("Training Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    // Eval
    printf("Autoencoder Train MSE = %.4f", autoencoder.eval(dataset));

    // Save model
    if (is_save_model)
        autoencoder.save_parameters(autoencoder_path);
}

template <typename AE>
Dataset phase_1_encode(const Dataset& dataset, const AE& autoencoder, const char *encoded_dataset_path,
                    bool is_save_encoded = true) {
    Dataset encoded_dataset = autoencoder.encode(dataset);
    if (is_save_encoded)
        write_data(encoded_dataset, encoded_dataset_path);
    return encoded_dataset;
}

// Phase 2: Train and evaluate SVM on encoded dataset
SVMmodel phase_2_train(const Dataset &encoded_dataset, const vector<int> &labels, const char* svm_model_path,
                    float train_ratio = TRAIN_RATIO, float c_param = C_PARAM, string kernel_type = string(KERNEL_PARAM),
                    string gamma_type = string(GAMMA_PARAM), float tolerance = TOLERANCE, float cache_size = CACHE_SIZE,
                    int max_iter = MAX_ITER, int nochange_steps = NOCHANGE_STEPS, int num_classes = NUM_CLASSES,
                    bool is_save_model = true) {
    vector<vector<double>> data;
    for (int i = 0; i < encoded_dataset.n; ++i) {
        vector<double> sample(encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth);
        for (int j = 0; j < sample.size(); ++j) {
            sample[j] = encoded_dataset.data[i * sample.size() + j];
        }
        data.push_back(sample);
    }

    // Split into train and test sets
    int train_size = static_cast<int>(train_ratio * encoded_dataset.n);
    vector<vector<double>> trainData(data.begin(), data.begin() + train_size);
    vector<int> trainLabels(labels.begin(), labels.begin() + train_size);
    vector<vector<double>> testData(data.begin() + train_size, data.end());
    vector<int> testLabels(labels.begin() + train_size, labels.end());

    // Train SVM model
    SVMmodel svm_model(c_param, kernel_type, gamma_type, tolerance, cache_size, max_iter, nochange_steps);
    svm_model.train(trainData, trainLabels);

    // Test SVM model
    vector<int> predictions = svm_model.predict(testData);
    double accuracy = svm_model.calculateAccuracy(predictions, testLabels, num_classes);
    vector<vector<int>> class_report = svm_model.calculateClassificationReport(predictions, testLabels, num_classes);
    vector<vector<int>> conf_matrix = svm_model.calculateConfusionMatrix(predictions, testLabels, num_classes);

    printf("SVM Train Accuracy: %.2f%%\n", accuracy * 100.0);
    svm_model.printClassificationReport(class_report);
    svm_model.printConfusionMatrix(conf_matrix);

    if (is_save_model) {
        svm_model.save(svm_model_path);
    }
    return svm_model;
}

SVMmodel phase_2_load(const char* svm_model_path) {
    SVMmodel svm_model;
    svm_model.load(svm_model_path);
    return svm_model;
}

double phase_2_test(SVMmodel& model, const Dataset &encoded_dataset, const vector<int> &labels,
                    int num_classes = NUM_CLASSES,
                    bool is_save_eval = true) {
    vector<vector<double>> data;
    for (int i = 0; i < encoded_dataset.n; ++i) {
        vector<double> sample(encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth);
        for (int j = 0; j < sample.size(); ++j) {
            sample[j] = encoded_dataset.data[i * sample.size() + j];
        }
        data.push_back(sample);
    }

    // Predict using SVM model
    vector<int> predictions = model.predict(data);
    double accuracy = model.calculateAccuracy(predictions, labels, num_classes);
    vector<vector<int>> class_report = model.calculateClassificationReport(predictions, labels, num_classes);
    vector<vector<int>> conf_matrix = model.calculateConfusionMatrix(predictions, labels, num_classes);

    printf("SVM Test Accuracy: %.2f%%\n", accuracy * 100.0);
    model.printClassificationReport(class_report);
    model.printConfusionMatrix(conf_matrix);

    if (is_save_eval) {
        // Save evaluation results to file
        const char *eval_file = "svm_evaluation.txt";
        ofstream ofs(eval_file);
        if (ofs.is_open()) {
            ofs << "SVM Test Accuracy: " << accuracy * 100.0 << "%\n";
            ofs << "Classification Report:\n";
            for (const auto& row : class_report) {
                for (const auto& val : row) {
                    ofs << val << " ";
                }
                ofs << "\n";
            }
            ofs << "Confusion Matrix:\n";
            for (const auto& row : conf_matrix) {
                for (const auto& val : row) {
                    ofs << val << " ";
                }
                ofs << "\n";
            }
            ofs.close();
            cout << "SVM evaluation results saved to " << eval_file << endl;
        } else {
            cerr << "Error: Cannot open file " << eval_file << " for writing." << endl;
        }
    }
    return accuracy;
}

// Visualization: Save training times and speedup data to CSV files
void save_bar_chart(const vector<string>& labels, const vector<double>& times, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }
    
    file << "Phase,Time(seconds)\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        file << labels[i] << "," << fixed << setprecision(3) << times[i] << "\n";
    }
    file.close();
    cout << "Bar chart data saved to " << filename << endl;
}

void save_speedup_graph(const vector<string>& labels, const vector<double>& speedups, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }
    
    file << "Phase,Speedup\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        file << labels[i] << "," << fixed << setprecision(3) << speedups[i] << "\n";
    }
    file.close();
    cout << "Speedup graph data saved to " << filename << endl;
}

int main(int argc, char *argv[]) {
    // Usage: ./main [RUN_MODE] [HARDWARE_MODE] [n_batches] [n_epoch] [batch_size] [learning_rate]
    int n_batches = NUM_BATCHES;
    int n_epoch = N_EPOCH;
    int batch_size = BATCH_SIZE;
    float learning_rate = LEARNING_RATE;
    if (argc > 1)   RUN_MODE = argv[1];
    if (argc > 2)   HARDWARE_MODE = argv[2];
    if (argc > 3)   n_batches = atoi(argv[3]);
    if (argc > 4)   n_epoch = atoi(argv[4]);
    if (argc > 5)   batch_size = atoi(argv[5]);
    if (argc > 6)   learning_rate = static_cast<float>(atof(argv[6]));

    bool    run_phase_1 = (string(RUN_MODE) == "phase_1" || string(RUN_MODE) == "all"),
            run_phase_2 = (string(RUN_MODE) == "phase_2" || string(RUN_MODE) == "all");

    cout << "Loading and preprocessing dataset..." << endl;
    Dataset dataset = load_dataset(DATASET_DIR, n_batches, true);

    Dataset encoded_dataset;
    if (run_phase_1) {
        cout << "Phase 1: Training Autoencoder and encoding dataset" << endl;
        if (HARDWARE_MODE == "cpu") {
            Cpu_Autoencoder autoencoder;
            phase_1_train<Cpu_Autoencoder>(dataset, autoencoder, AUTOENCODER_OUTPUT_DIR, CPU_AUTOENCODER_PATH,
                                            n_epoch, batch_size, learning_rate, VERBOSE, CHECKPOINT);
            encoded_dataset = phase_1_encode(dataset, autoencoder, ENCODED_DATASET_PATH);
        } else {
            Gpu_Autoencoder autoencoder;
            phase_1_train<Gpu_Autoencoder>(dataset, autoencoder, AUTOENCODER_OUTPUT_DIR, GPU_AUTOENCODER_PATH,
                                            n_epoch, batch_size, learning_rate, VERBOSE, CHECKPOINT);
            encoded_dataset = phase_1_encode(dataset, autoencoder, ENCODED_DATASET_PATH);
        }
    } else {
        cout << "Skipping Phase 1: Loading trained Autoencoder and encoding dataset" << endl;
        if (HARDWARE_MODE == "cpu") {
            Cpu_Autoencoder autoencoder;
            autoencoder.load_parameters(CPU_AUTOENCODER_PATH);
            encoded_dataset = phase_1_encode(dataset, autoencoder, ENCODED_DATASET_PATH);
        } else {
            Gpu_Autoencoder autoencoder;
            autoencoder.load_parameters(GPU_AUTOENCODER_PATH);
            encoded_dataset = phase_1_encode(dataset, autoencoder, ENCODED_DATASET_PATH);
        }
    }
    cout << "Encoded dataset has " << encoded_dataset.n << " samples, each of size "
         << encoded_dataset.width << "x" << encoded_dataset.height << "x" << encoded_dataset.depth << endl;

    vector<int> labels;
    if (run_phase_1) {
        cout << "Loading labels from file" << endl;
        const char *labels_path = (string(DATASET_DIR) + "/" + string(LABELS_FILE)).c_str();
        labels = load_labels(labels_path, NUM_TRAIN_SAMPLES);
    } else {
        cout << "Using dummy labels for testing" << endl;
        labels = dummy_labels(NUM_TRAIN_SAMPLES, NUM_CLASSES);
    }

    SVMmodel svm_model;
    if (run_phase_2) {
        cout << "Phase 2: Training SVM on Encoded Data" << endl;
        svm_model = phase_2_train(encoded_dataset, labels, SVM_MODEL_FILE,
                      TRAIN_RATIO, C_PARAM, string(KERNEL_PARAM), string(GAMMA_PARAM),
                      TOLERANCE, CACHE_SIZE, MAX_ITER, NOCHANGE_STEPS, NUM_CLASSES);
    } else {
        cout << "Skipping Phase 2: Loading trained SVM model" << endl;
        svm_model = phase_2_load(SVM_MODEL_FILE);
    }

    cout << "Testing SVM on Encoded Data" << endl;
    double test_accuracy = phase_2_test(svm_model, encoded_dataset, labels, NUM_CLASSES);
    cout << "SVM Test Accuracy on Encoded Data: " << test_accuracy * 100.0 << "%" << endl;

    return 0;
}