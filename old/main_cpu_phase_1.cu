#include <cstdio>
#include "cpu_autoencoder.h"
#include "data_loader.h"

int main(int argc, char *argv[]) {
  // Read dataset
  const char *dataset_dir = "./data/cifar-10-batches-bin";
  bool is_train = true;
  Dataset dataset = load_dataset(dataset_dir, is_train);

  // Shuffle dataset
  shuffle_dataset(dataset);

  // Create and train model
  Cpu_Autoencoder autoencoder;
  int n_epoch = 20;
  int batch_size = 32;
  float learning_rate = 0.001f;
  bool verbose = false;
  int checkpoint = 0;
  const char *output_dir = "./model";
  autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, seed, checkpoint, output_dir);

  // Eval
  printf("Autoencoder MSE = %.4f", autoencoder.eval(dataset));
}