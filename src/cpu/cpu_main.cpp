#include "cpu_autoencoder.h"
#include "cpu_data_loader.h"

#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char *argv[]) {
    const char *dataset_dir = "./data/cifar-10-batches-bin";
    const char *output_dir  = "./output";
    const char *model_file  = "./cpu_autoencoder_model.bin";
    const char *encoded_file = "./encoded_dataset.bin";

    // Defaults
    int n_batches = 5;   // CIFAR-10 has 5 training batches
    int n_epoch = 20;
    int batch_size = 32;
    float learning_rate = 0.001f;
    bool verbose = false;
    int checkpoint = 0;

    // CLI overrides: ./cpu_cpp_main.exe [n_batches] [n_epoch] [batch_size] [learning_rate] [verbose] [checkpoint]
    if (argc > 1) n_batches     = atoi(argv[1]);
    if (argc > 2) n_epoch       = atoi(argv[2]);
    if (argc > 3) batch_size    = atoi(argv[3]);
    if (argc > 4) learning_rate = static_cast<float>(atof(argv[4]));
    if (argc > 5) verbose       = atoi(argv[5]) != 0;
    if (argc > 6) checkpoint    = atoi(argv[6]);

    // Read dataset
    bool is_train = true;
    Dataset dataset = load_dataset(dataset_dir, n_batches, is_train);

    // Shuffle dataset
    shuffle_dataset(dataset);

    // Create and train model
    Cpu_Autoencoder autoencoder;
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    // Eval
    printf("CPU Autoencoder MSE = %.4f", autoencoder.eval(dataset));

    // Save model
    autoencoder.save_parameters(model_file);

    // Save encoded dataset
    Dataset encoded_dataset = autoencoder.encode(dataset);
    write_data(encoded_dataset, encoded_file);
}