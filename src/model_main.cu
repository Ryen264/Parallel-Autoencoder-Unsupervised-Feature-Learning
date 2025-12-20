#include "constants.h"
#include "data_loader.h"
#include "model.h"

#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>
using namespace std;

// string RUN_MODE       = "phase_2";  // "phase_1", "phase_2", "all"
// string HARDWARE_MODE  = "cpu";  // "cpu", "gpu"
bool USE_DUMMY_DATA   = true;  // only for phase 2
bool IS_SAVE_MODEL    = true;

// const string DATASET_DIR            = "./data/cifar-10-batches-bin";
const string DATASET_DIR            = "/content/data/cifar-10-batches-bin";
// const string MODEL_OUTPUT_DIR       = "./model";
const string MODEL_OUTPUT_DIR       = "/content/model";

const string ENCODED_DATASET_FILE   = "encoded_dataset.bin";
const string LABELS_FILE            = "labels.bin";
const string SVM_MODEL_FILE         = "svm_model.bin";

const string VISUALIZATION_TRAINING_TIMES_SVG = "training_times.svg";
const string VISUALIZATION_TRAINING_TIMES_CSV = "training_times.csv";
const string VISUALIZATION_SPEEDUP_GRAPH_SVG  = "speedup_graph.svg";
const string VISUALIZATION_SPEEDUP_GRAPH_CSV  = "speedup_data.csv";
const string VISUALIZATION_HTML_DASHBOARD     = "performance_analysis.html";

Dataset dummy_dataset(int n, int width, int height, int depth) {
    unique_ptr<float[]> data(new float[n * width * height * depth]);
    for (int i = 0; i < n * width * height * depth; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Dataset(data, n, width, height, depth);
}

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