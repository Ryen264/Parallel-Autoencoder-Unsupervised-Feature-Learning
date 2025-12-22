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
// bool USE_DUMMY_DATA   = true;  // only for phase 2
bool IS_SAVE_MODEL    = true;

// const string DATASET_DIR            = "./data/cifar-10-batches-bin";
const string DATASET_DIR            = "/content/data/cifar-10-batches-bin";
// const string MODEL_OUTPUT_DIR       = "./model";
const string MODEL_OUTPUT_DIR       = "/content/model";

const string ENCODED_DATASET_FILE   = "encoded_dataset.bin";
const string LABELS_FILE            = "labels.bin";
const string SVM_MODEL_FILE         = "svm_model.bin";

Dataset dummy_dataset(int n, int width, int height, int depth) {
    unique_ptr<float[]> data(new float[n * width * height * depth]);
    for (int i = 0; i < n * width * height * depth; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Dataset(data, n, width, height, depth);
}

SVMmodel phase_2_train(const Dataset &encoded_dataset, const vector<int> &labels, float train_ratio = 0.8f, bool is_save_model = true) {
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

    return svm_model;
}

double phase_2_test(SVMmodel model, const Dataset &encoded_dataset, const vector<int> &labels) {
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
    double accuracy = model.calculateAccuracy(predictions, labels, NUM_CLASSES);
    vector<vector<int>> class_report = model.calculateClassificationReport(predictions, labels, NUM_CLASSES);
    vector<vector<int>> conf_matrix = model.calculateConfusionMatrix(predictions, labels, NUM_CLASSES);

    printf("SVM Test Accuracy: %.2f%%\n", accuracy * 100.0);
    model.printClassificationReport(class_report);
    model.printConfusionMatrix(conf_matrix);

    return accuracy;
}

int main(int argc, char *argv[]) {
    cout << "Phase 2: Training SVM on Encoded Data" << endl;
    
    int num_train_samples = 50000;
    int num_test_samples = 10000;

    Dataset train_encoded_dataset, test_encoded_dataset;
    vector<int> train_labels, test_labels;
    
    train_encoded_dataset = dummy_dataset(num_train_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
    test_encoded_dataset = dummy_dataset(num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

    // Create dummy labels
    train_labels.resize(num_train_samples);
    for (int i = 0; i < num_train_samples; ++i) {
        train_labels[i] = rand() % NUM_CLASSES; // 10 classes
    }
    test_labels.resize(num_test_samples);
    for (int i = 0; i < num_test_samples; ++i) {
        test_labels[i] = rand() % NUM_CLASSES; // 10 classes
    }

    // Train SVM on encoded data
    SVMmodel svm_model = phase_2_train(train_encoded_dataset, train_labels, TRAIN_RATIO, IS_SAVE_MODEL);

    // Test SVM on encoded data
    double test_accuracy = phase_2_test(svm_model, test_encoded_dataset, test_labels);
    return 0;
}