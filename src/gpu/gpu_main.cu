#include "constants.h"
#include "data_loader.h"
#include "gpu_autoencoder.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
using namespace std;

constexpr const char *DATASET_DIR          = "./data/cifar-10-batches-bin";
constexpr const char *MODEL_OUTPUT_DIR     = "./model";
constexpr const char *ENCODED_DATASET_DIR  = "./data/encoded";
constexpr const char *ENCODED_DATASET_FILE = "./data/encoded/gpu_encoded_dataset.bin";
constexpr const char *ENCODED_LABEL_FILE   = "./data/encoded/gpu_label_dataset.bin";
constexpr int         N_BATCHES            = 2;

// Dataset phase_1_gpu(const char *dataset_dir,
//                     const char *output_dir,
//                     bool        is_train      = true,
//                     int         n_batches     = 2,
//                     int         n_epoch       = 20,
//                     int         batch_size    = 32,
//                     float       learning_rate = 0.001f,
//                     bool        verbose       = false,
//                     int         checkpoint    = 0) {
//   Dataset dataset = load_dataset(dataset_dir, n_batches, is_train);
//   shuffle_dataset(dataset);

//   Gpu_Autoencoder autoencoder;
//   printf(
//       "Training GPU Autoencoder for %d epochs with batch size %d and learning rate "
//       "%.4f\n",
//       n_epoch,
//       batch_size,
//       learning_rate);
//   autoencoder.fit(
//       dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

//   return autoencoder.encode(dataset);
// }

int main() {
  Gpu_Autoencoder autoencoder;
  Dataset         dataset = load_dataset(DATASET_DIR, N_BATCHES, true);

  autoencoder.fit(dataset,
                  N_EPOCH,
                  BATCH_SIZE,
                  LEARNING_RATE,
                  VERBOSE,
                  CHECKPOINT,
                  MODEL_OUTPUT_DIR);

  puts("\n=======================ENCODING TEST DATASET=======================");
  filesystem::create_directories(ENCODED_DATASET_DIR);
  dataset = load_dataset(DATASET_DIR, 1, false);

  Dataset encoded_dataset      = autoencoder.encode(dataset);
  int     encoded_dataset_size = encoded_dataset.n *
                             encoded_dataset.width *
                             encoded_dataset.height *
                             encoded_dataset.depth *
                             sizeof(float);
  ofstream buffer(ENCODED_DATASET_FILE, ios::out | ios::binary);
  buffer.write(encoded_dataset.get_data(), encoded_dataset_size);
  buffer.close();

  buffer.open(ENCODED_LABEL_FILE, ios::out | ios::binary);
  buffer.write(encoded_dataset.get_labels(), encoded_dataset.n * sizeof(int));
  buffer.close();
}