#include "data_loader.h"
#include "gpu_autoencoder.h"

Dataset phase_1_gpu(const char *dataset_dir, const char *output_dir,
                    bool is_train = true, int n_epoch = 20, int batch_size = 32, float learning_rate = 0.001f, bool verbose = false, int checkpoint = 0) {
    // Read dataset
    Dataset dataset = load_dataset(dataset_dir, is_train);

    // Shuffle dataset
    shuffle_dataset(dataset);

    // Create and train model
    Gpu_Autoencoder autoencoder;
    printf("Training GPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    // Eval
    printf("GPU Autoencoder MSE = %.4f", autoencoder.eval(dataset));
    return autoencoder.encode(dataset);
}

int main() {
  Dataset dataset = load_dataset("data");

  Gpu_Autoencoder autoencoder;
  autoencoder.fit(dataset, 20, 32, 0.001, true, 0);
}