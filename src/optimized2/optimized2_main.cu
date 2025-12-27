#include "constants.h"
#include "optimized2_autoencoder.h"
#include "optimized_data_loader.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
using namespace std;

constexpr const char *DATASET_DIR         = "./data/cifar-10-batches-bin";
constexpr const char *MODEL_OUTPUT_DIR    = "./model";
constexpr const char *ENCODED_DATASET_DIR = "./data/encoded";
constexpr const char *ENCODED_DATASET_FILE =
    "./data/encoded/optimized2_encoded_dataset.bin";
constexpr const char *ENCODED_LABEL_FILE =
    "./data/encoded/optimized2_label_dataset.bin";
constexpr int N_BATCHES = 2;

int main() {
  Optimized2_Autoencoder autoencoder;
  Optimized_Dataset      dataset = read_optimized_dataset(DATASET_DIR, N_BATCHES, true);

  autoencoder.fit(
      dataset, N_EPOCH, BATCH_SIZE, LEARNING_RATE, CHECKPOINT, MODEL_OUTPUT_DIR);

  puts("\n=======================ENCODING TRAIN DATASET=======================");
  filesystem::create_directories(ENCODED_DATASET_DIR);
  dataset = autoencoder.encode(dataset);

  int encoded_dataset_size =
      dataset.n * dataset.width * dataset.height * dataset.depth * sizeof(float);
  ofstream buffer(ENCODED_DATASET_FILE, ios::out | ios::binary);
  buffer.write(reinterpret_cast<char *>(dataset.data), encoded_dataset_size);
  buffer.close();

  buffer.open(ENCODED_LABEL_FILE, ios::out | ios::binary);
  buffer.write(reinterpret_cast<char *>(dataset.labels), dataset.n * sizeof(int));
  buffer.close();

  puts("\n=======================ENCODING TEST DATASET=======================");
  dataset = autoencoder.encode(read_optimized_dataset(DATASET_DIR, 1, false));

  encoded_dataset_size =
      dataset.n * dataset.width * dataset.height * dataset.depth * sizeof(float);
  buffer.open(ENCODED_DATASET_FILE, ios::out | ios::binary);
  buffer.write(reinterpret_cast<char *>(dataset.data), encoded_dataset_size);
  buffer.close();

  buffer.open(ENCODED_LABEL_FILE, ios::out | ios::binary);
  buffer.write(reinterpret_cast<char *>(dataset.labels), dataset.n * sizeof(int));
  buffer.close();
}