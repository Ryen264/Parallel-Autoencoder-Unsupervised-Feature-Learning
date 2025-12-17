#include "constants.h"
#include "data_loader.h"
#include "gpu_autoencoder.h"

int main() {
  Dataset dataset = load_dataset("data/cifar-10-batches-bin", 2);

  Gpu_Autoencoder autoencoder;
  autoencoder.fit(dataset, N_EPOCH, BATCH_SIZE, LEARNING_RATE, VERBOSE, CHECKPOINT);
}