#include "cpu_autoencoder.h"
#include "dataset.h"
#include <cstdio>

int main(int argc, char *argv[]) {
  // Parse args...
  const char *dataset_dir = "./data";

  // Read dataset
  Dataset dataset = load_dataset(dataset_dir);

  // Shuffle dataset
  shuffle_dataset(dataset);

  // Create and train model
  Cpu_Autoencoder autoencoder;
  autoencoder.fit(dataset, 20, 32, 0.001, 0, 0);

  // Eval
  printf("Autoencoder MSE = %.4f", autoencoder.eval(dataset));
}