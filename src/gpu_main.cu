#include "data_loader.h"
#include "gpu_autoencoder.h"

int main() {
  Dataset dataset = load_dataset("data");

  Gpu_Autoencoder autoencoder;
  autoencoder.fit(dataset, 20, 32, 0.001, true, 0);
}