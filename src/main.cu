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

string RUN_MODE       = "all";  // "phase_1", "phase_2", "all"
string HARDWARE_MODE  = "gpu";  // "cpu", "gpu"

// NUM_BATCHES      = 1;  // CIFAR-10 has 5 training batches
// N_EPOCH          = 20;
// BATCH_SIZE       = 32;
// LEARNING_RATE    = 0.001f;
// VERBOSE          = false;
// CHECKPOINT       = 0;

const char *DATASET_DIR             = "./data/cifar-10-batches-bin";
const char *AUTOENCODER_OUTPUT_DIR  = "./output";
const char *CPU_AUTOENCODER_PATH    = "./cpu_autoencoder_model.bin";
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
Dataset load_dataset(const char *dataset_dir, int n_batches = -1, bool is_train = true) {
    if (n_batches <= 0) {
        n_batches = N_BATCHES;
    }

    // Read dataset
    Dataset dataset = read_dataset(dataset_dir, n_batches, is_train);

    // Shuffle dataset
    shuffle_dataset(dataset);
    return dataset;
}

// Load labels from file
vector<int> load_labels(const char *labels_path, int n_samples = -1) {
    if (n_samples <= 0) {
        n_samples = NUM_TRAIN_SAMPLES;
    }

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
Dataset dummy_dataset(int n, int width, int height, int depth) {
    unique_ptr<float[]> data(new float[n * width * height * depth]);
    for (int i = 0; i < n * width * height * depth; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Dataset(data, n, width, height, depth);
}

// Create dummy labels for phase 2 testing
vector<int> dummy_labels(int n = -1, int num_classes = -1) {
    if (n <= 0) {
        n = NUM_TRAIN_SAMPLES;
    }
    if (num_classes <= 0) {
        num_classes = NUM_CLASSES;
    }

    vector<int> labels(n);
    for (int i = 0; i < n; ++i) {
        labels[i] = rand() % num_classes;
    }
    return labels;
}

// Phase 1: Train and evaluate CPU autoencoder
Dataset phase_1_cpu(Dataset dataset, const char *output_dir, const char *autoencoder_path, const char *encoded_dataset_path,
                    int n_epoch = -1, int batch_size = -1, float learning_rate = -0.001f, bool verbose = false, int checkpoint = 0,
                    bool is_save_model = true, bool is_save_encoded = true) {
    if (n_epoch <= 0) {
        n_epoch = N_EPOCH;
    }
    if (batch_size <= 0) {
        batch_size = BATCH_SIZE;
    }
    if (learning_rate <= 0.0f) {
        learning_rate = LEARNING_RATE;
    }

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

// Phase 1: Train and evaluate GPU autoencoder
Dataset phase_1_gpu(Dataset dataset, const char *output_dir, , const char *autoencoder_path, const char *encoded_dataset_path,
                    int n_epoch = 20, int batch_size = 32, float learning_rate = 0.001f, bool verbose = false, int checkpoint = 0,
                    bool is_save_model = true, bool is_save_encoded = true) {
    if (n_epoch <= 0) {
        n_epoch = N_EPOCH;
    }
    if (batch_size <= 0) {
        batch_size = BATCH_SIZE;
    }
    if (learning_rate <= 0.0f) {
        learning_rate = LEARNING_RATE;
    }

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

// Phase 2: Train and evaluate SVM on encoded dataset
double phase_2(const Dataset &encoded_dataset, const vector<int> &labels, float train_ratio = 0.8f, bool is_save_model = true) {
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
    SVMmodel svm_model(C, KERNEL_TYPE, GAMMA_TYPE, TOLERANCE, CACHE_SIZE, MAX_ITER, NOCHANGE_STEPS);
    svm_model.train(trainData, trainLabels);

    // Test SVM model
    vector<int> predictions = svm_model.predict(testData);
    double accuracy = svm_model.calculateAccuracy(testLabels, predictions, NUM_CLASSES);
    vector<double> conf_matrix = svm_model.calculateConfusionMatrix(testLabels, predictions, NUM_CLASSES);

    svm_model.printAccuracy(accuracy);
    svm_model.printConfusionMatrix(conf_matrix);

    if (is_save_model) {
        string svm_model_path = MODEL_OUTPUT_DIR + "/" + SVM_MODEL_FILE;
        svm_model.save(svm_model_path);
    }

    return accuracy;
}

/*
double phase_2(const Dataset &encoded_dataset, const vector<int> &labels, float train_ratio = 0.8f, bool is_save_model = true) {
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
    SVMmodel svm_model(C_PARAM, KERNEL_PARAM, GAMMA_PARAM, TOLERANCE, CACHE_SIZE, MAX_ITER, NOCHANGE_STEPS);
    svm_model.train(trainData, trainLabels);

    // Test SVM model
    vector<int> predictions = svm_model.predict(testData);
    double accuracy = svm_model.calculateAccuracy(predictions, testLabels, NUM_CLASSES);
    vector<vector<int>> class_report = svm_model.calculateClassificationReport(predictions, testLabels, NUM_CLASSES);
    vector<vector<int>> conf_matrix = svm_model.calculateConfusionMatrix(predictions, testLabels, NUM_CLASSES);

    printf("SVM Accuracy: %.2f%%\n", accuracy * 100.0);
    svm_model.printClassificationReport(class_report);
    svm_model.printConfusionMatrix(conf_matrix);

    if (is_save_model) {
        string svm_model_path = MODEL_OUTPUT_DIR + "/" + SVM_MODEL_FILE;
        svm_model.save(svm_model_path);
    }

    return accuracy;
}

int main(int argc, char *argv[]) {
    cout << "Phase 2: Training SVM on Encoded Data" << endl;
    
    Dataset encoded_dataset;
    vector<int> labels;
    
    if (USE_DUMMY_DATA) {
        // Use dummy data for phase 2
        cout << "Using dummy data for Phase 2" << endl;
        int num_samples = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES;

        // actual dataset
        int num_train_samples = 50000;
        int num_test_samples = 10000;
        num_samples = num_train_samples + num_test_samples;

        encoded_dataset = dummy_dataset(num_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
        
        // Create dummy labels
        labels.resize(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            labels[i] = rand() % NUM_CLASSES; // 10 classes
        }
    } else {
        // Load encoded dataset from phase 1
        cout << "Loading encoded dataset from phase 1" << endl;
        FILE *f = fopen(ENCODED_DATASET_FILE.c_str(), "rb");
        if (!f) {
            cerr << "Error: Encoded dataset file not found!" << endl;
            cerr << "To use real data, make sure to run Phase 1 first or set USE_DUMMY_DATA = true" << endl;
            return -1;
        }
        
        int num_samples = NUM_TRAIN_SAMPLES;  // Local variable for actual data size
        unique_ptr<float[]> encoded_data(new float[num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH]);
        size_t bytes_read = fread(encoded_data.get(), sizeof(float), num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH, f);
        fclose(f);
        
        if (bytes_read != num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH) {
            cerr << "Warning: Expected " << (num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH) 
                        << " elements, but read " << bytes_read << endl;
            num_samples = bytes_read / (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH);
        }
        
        encoded_dataset = Dataset(encoded_data, num_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

        // Load labels from phase 1
        labels.resize(NUM_TRAIN_SAMPLES);
        string labels_path = DATASET_DIR + "/" + LABELS_FILE;
        FILE *lf = fopen(labels_path.c_str(), "rb");
        if (lf) {
            size_t labels_read = fread(labels.data(), sizeof(int), NUM_TRAIN_SAMPLES, lf);
            if (labels_read != NUM_TRAIN_SAMPLES) {
                cerr << "Warning: Expected " << NUM_TRAIN_SAMPLES << " labels, but read " << labels_read << endl;
            }
            fclose(lf);
        } else {
            cerr << "Warning: Labels file not found, using random labels" << endl;
            return -1;
        }
    }

    // Train SVM on encoded data
    float train_ratio = NUM_TRAIN_SAMPLES / static_cast<float>(NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES);

    // actual ratio
    train_ratio = 50000 / static_cast<float>(50000 + 10000);

    double accuracy = phase_2(encoded_dataset, labels, train_ratio, IS_SAVE_MODEL);
    return 0;
}
*/

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
    if (argc > 1)   RUN_MODE = argv[1];
    if (argc > 2)   HARDWARE_MODE = argv[2];
    if (argc > 3)   N_BATCHES = atoi(argv[3]);
    if (argc > 4)   N_EPOCH = atoi(argv[4]);
    if (argc > 5)   BATCH_SIZE = atoi(argv[5]);
    if (argc > 6)   LEARNING_RATE = atof(argv[6]);

    bool    run_phase_1 = (string(RUN_MODE) == "phase_1" || string(RUN_MODE) == "all"),
            run_phase_2 = (string(RUN_MODE) == "phase_2" || string(RUN_MODE) == "all"),

    Dataset encoded_dataset;
    if (run_phase_1) {
        cout << "Loading and preprocessing dataset..." << endl;
        Dataset dataset = load_data(DATASET_DIR, N_BATCHES, true);

        cout << "Phase 1: Training Autoencoder" << endl;
        if (HARDWARE_MODE == "cpu") {
            encoded_dataset = phase_1_cpu(dataset, AUTOENCODER_OUTPUT_DIR, CPU_AUTOENCODER_PATH, ENCODED_DATASET_PATH,
                                            N_EPOCH, BATCH_SIZE, LEARNING_RATE, VERBOSE, CHECKPOINT);
        } else {
            encoded_dataset = phase_1_gpu(DATASET_DIR.c_str(), AUTOENCODER_OUTPUT_DIR.c_str(), is_train,
                                            N_EPOCH, BATCH_SIZE, LEARNING_RATE, VERBOSE, CHECKPOINT);
        }        
    } else {
        cout << "Skipping Phase 1: Loading encoded dataset from file" << endl;
        encoded_dataset = dummy_dataset(NUM_TRAIN_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
    }

    if (run_phase_2) {
        cout << "Phase 2: Training SVM on Encoded Data" << endl;
        vector<int> labels;
        if (run_phase_1) {
            // Load labels from phase 1
            const char *labels_path = (string(DATASET_DIR) + "/" + string(LABELS_FILE)).c_str();
            labels = load_labels(labels_path, NUM_TRAIN_SAMPLES);
        } else {
            // Create dummy labels
            labels = dummy_labels(NUM_TRAIN_SAMPLES, NUM_CLASSES);
        }

        // Train SVM on encoded data
        double accuracy = phase_2(encoded_dataset, labels, TRAIN_RATIO, IS_SAVE_MODEL);
                
        cout << "SVM Accuracy on Encoded Data: " << accuracy * 100.0 << "%" << endl;
        cout << "Phase 2 completed in " << fixed << setprecision(2) << phase2_time << " seconds" << endl;
    }

    return 0;
}