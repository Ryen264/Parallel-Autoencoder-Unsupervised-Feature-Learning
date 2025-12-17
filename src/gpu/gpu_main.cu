#include "constants.h"
#include "data_loader.h"
#include "gpu_autoencoder.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>
using namespace std;

constexpr string_view DATASET_DIR          = "./data/cifar-10-batches-bin";
constexpr string_view MODEL_OUTPUT_DIR     = "./model";
constexpr string_view ENCODED_DATASET_DIR  = "./data/encoded";
constexpr string_view ENCODED_DATASET_FILE = "./data/encoded/gpu_encoded_dataset.bin";
constexpr string_view ENCODED_LABEL_FILE   = "./data/encoded/gpu_label_dataset.bin";
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
  Dataset         dataset = load_dataset(DATASET_DIR.data(), N_BATCHES, true);

  autoencoder.fit(train_dataset,
                  N_EPOCH,
                  BATCH_SIZE,
                  LEARNING_RATE,
                  VERBOSE,
                  CHECKPOINT,
                  MODEL_OUTPUT_DIR);

  puts("\n=======================ENCODING TEST DATASET=======================");
  filesystem::create_directories(ENCODED_DATASET_DIR);
  dataset = load_dataset(DATASET_DIR.data(), 1, false);

  Dataset encoded_dataset      = autoencoder.encode(dataset);
  int     encoded_dataset_size = encoded_dataset.n *
                             encoded_dataset.width *
                             encoded_dataset.height *
                             encoded_dataset.depth *
                             sizeof(float);
  ofstream buffer(ENCODED_DATASET_FILE, "wb");
  buffer.write(encoded_dataset.get_data(), encoded_dataset_size);
  buffer.close();

  buffer.open(ENCODED_LABEL_FILE, "wb");
  buffer.write(encoded_dataset.get_labels(), encoded_dataset.n * sizeof(int));
  buffer.close();
}