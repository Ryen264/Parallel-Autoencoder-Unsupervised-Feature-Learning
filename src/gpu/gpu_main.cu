#include "constants.h"
#include "data_loader.h"
#include "gpu_autoencoder.h"

#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>
using namespace std;

// string RUN_MODE      = "phase_1";// "phase_1", "phase_2", "all"
// string HARDWARE_MODE = "gpu";    // "cpu", "gpu"
// bool USE_DUMMY_DATA  = false;    // only for phase 2
// bool IS_SAVE_MODEL   = true;

const string DATASET_DIR       = "./data/cifar-10-batches-bin";
const string MODEL_OUTPUT_DIR  = "./model";

const string ENCODED_DATASET_FILE = "encoded_dataset.bin";
const string LABELS_FILE          = "labels.bin";
const string SVM_MODEL_FILE       = "svm_model.bin";

const string VISUALIZATION_TRAINING_TIMES_SVG = "training_times.svg";
const string VISUALIZATION_TRAINING_TIMES_CSV = "training_times.csv";
const string VISUALIZATION_SPEEDUP_GRAPH_SVG  = "speedup_graph.svg";
const string VISUALIZATION_SPEEDUP_GRAPH_CSV  = "speedup_data.csv";
const string VISUALIZATION_HTML_DASHBOARD     = "performance_analysis.html";

Dataset phase_1_gpu(const char *dataset_dir, const char *output_dir,
                    bool is_train = true, int n_epoch = 20, int batch_size = 32, float learning_rate = 0.001f, bool verbose = false, int checkpoint = 0) {
    Dataset dataset = load_dataset(dataset_dir, is_train);
    shuffle_dataset(dataset);

    Gpu_Autoencoder autoencoder;
    printf("Training GPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    return autoencoder.encode(dataset);
}

int main(int argc, char *argv[]) {
    // Timing variables
    vector<string> phase_labels;
    vector<double> phase_times;
    double baseline_time = 0.0;
    vector<double> speedups;

    cout << "Phase 1: Training Autoencoder" << endl;
    
    bool is_train = true;
    Dataset encoded_dataset = phase_1_gpu(DATASET_DIR.c_str(), MODEL_OUTPUT_DIR.c_str(), is_train,
                                        N_EPOCH, BATCH_SIZE, LEARNING_RATE, VERBOSE, CHECKPOINT);
    
    // Save encoded dataset for phase 2
    FILE *f = fopen(ENCODED_DATASET_FILE.c_str(), "wb");
    fwrite(encoded_dataset.data.get(), sizeof(float), encoded_dataset.n * encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth, f);
    fclose(f);
    return 0;
}