#include "constants.h"
#include "data_loader.h"
#include "cpu_autoencoder.h"

#include <iostream>
using namespace std;

// Cách chạy:
// ./main [phase_1_mode=train/load] \
//        [n_batches] [n_epoch] [batch_size] [learning_rate]

// Test by just using some first samples
int TRAIN_SAMPLES = -1;
int TEST_SAMPLES = -1;

const char *DATASET_DIR             = "./data/cifar-10-batches-bin";
const char *OUTPUT_DIR              = "./output";
const char *CPU_AUTOENCODER_PATH    = "./model/cpu_autoencoder_model.bin";
const char *ENCODED_DATASET_PATH    = "./output/encoded_dataset.bin";

// Load and preprocess dataset
Dataset load_dataset(const char *dataset_dir = DATASET_DIR, int n_batches = NUM_BATCHES, bool is_train = true) {
    Dataset dataset = read_dataset(dataset_dir, n_batches, is_train);
    shuffle_dataset(dataset);
    return dataset;
}

// Phase 1: Train and evaluate Autoencoder on trainset
template <typename AE>
AE phase_1_train(const Dataset& dataset, const char *output_dir = OUTPUT_DIR, const char *autoencoder_path = CPU_AUTOENCODER_PATH,
                    int n_epoch = N_EPOCH, int batch_size = BATCH_SIZE, float learning_rate = LEARNING_RATE, int checkpoint = CHECKPOINT,
                    bool is_save_model = true) {
    AE autoencoder;
    printf("Training CPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, checkpoint, output_dir);

    // Eval
    printf("Autoencoder Train MSE = %.4f\n", autoencoder.eval(dataset));

    // Save model
    if (is_save_model)
        autoencoder.save_parameters(autoencoder_path);

    return autoencoder;
}

template <typename AE>
AE phase_1_load(const char *autoencoder_path = CPU_AUTOENCODER_PATH) {
    AE autoencoder;
    autoencoder.load_parameters(autoencoder_path);
    printf("Loaded CPU Autoencoder model from %s\n", autoencoder_path);
    return autoencoder;
}

// Phase 1: Encode dataset using trained Autoencoder
template <typename AE>
Dataset phase_1_encode(const Dataset& dataset, const AE& autoencoder, const char *encoded_dataset_path = ENCODED_DATASET_PATH,
                    bool is_save_encoded = true) {
    Dataset encoded_dataset = autoencoder.encode(dataset);
    printf("Encoded dataset: n=%d, width=%d, height=%d, depth=%d\n",
           encoded_dataset.n, encoded_dataset.width, encoded_dataset.height, encoded_dataset.depth);

    if (is_save_encoded)
        write_binary(encoded_dataset, encoded_dataset_path);

    return encoded_dataset;
}

int main(int argc, char *argv[]) {
    bool train_phase_1 = true;

    int n_batches = NUM_BATCHES;
    int n_epoch = N_EPOCH;
    int batch_size = BATCH_SIZE;
    float learning_rate = LEARNING_RATE;

    if (argc > 1)   train_phase_1 = (string(argv[1]) == "train") ? true : false;
    if (argc > 2)   n_batches = atoi(argv[2]);
    if (argc > 3)   n_epoch = atoi(argv[3]);
    if (argc > 4)   batch_size = atoi(argv[4]);
    if (argc > 5)   learning_rate = atof(argv[5]);
    
    cout << "Loading and preprocessing datasets..." << endl;
    Dataset trainset = load_dataset(DATASET_DIR, n_batches, true);
    Dataset testset = load_dataset(DATASET_DIR, 1, false);

    // Test by just using some first samples
    if (TRAIN_SAMPLES > 0)
        trainset.n = TRAIN_SAMPLES;
    if (TEST_SAMPLES > 0)
        testset.n = TEST_SAMPLES;

    Dataset encoded_trainset, encoded_testset;

    // --- CHỈ CHẠY CPU AUTOENCODER ---
    Cpu_Autoencoder cpu_autoencoder;
    if (train_phase_1) {
        cpu_autoencoder = phase_1_train<Cpu_Autoencoder>(trainset, OUTPUT_DIR, CPU_AUTOENCODER_PATH,
                                                        n_epoch, batch_size, learning_rate, CHECKPOINT,
                                                        true);
    } else {
        cpu_autoencoder = phase_1_load<Cpu_Autoencoder>(CPU_AUTOENCODER_PATH);
    }
    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using CPU Autoencoder...\n");
    encoded_trainset = phase_1_encode<Cpu_Autoencoder>(trainset, cpu_autoencoder, ENCODED_DATASET_PATH,
                                                    true);
    encoded_testset = phase_1_encode<Cpu_Autoencoder>(testset, cpu_autoencoder, ENCODED_DATASET_PATH,
                                                    false);

    cout << "Process finished. SVM phase skipped." << endl;

    return 0;
}