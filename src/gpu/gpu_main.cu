#include "constants.h"
#include "data_loader.h"
#include "gpu_autoencoder.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
using namespace std;

constexpr const char *DATASET_DIR          = "/content/drive/MyDrive/@fithcmú/LapTrinhSongSong/data/cifar-10-batches-bin";
constexpr const char *MODEL_OUTPUT_DIR     = "/content/drive/MyDrive/@fithcmú/LapTrinhSongSong/model";
constexpr const char *ENCODED_DATASET_DIR  = "/content/drive/MyDrive/@fithcmú/LapTrinhSongSong/data/encoded";
constexpr const char *ENCODED_DATASET_FILE = "/content/drive/MyDrive/@fithcmú/LapTrinhSongSong/data/encoded/gpu_encoded_dataset.bin";
constexpr const char *ENCODED_LABEL_FILE   = "/content/drive/MyDrive/@fithcmú/LapTrinhSongSong/data/encoded/gpu_label_dataset.bin";
constexpr int         N_BATCHES            = 2;

int main() {
  Gpu_Autoencoder autoencoder;
  Dataset         dataset = load_dataset(DATASET_DIR, N_BATCHES, true);

  // check dataset loaded by showing first 3 images and their labels
  printf("Dataset loaded: %d images, %dx%dx%d\n", dataset.n, dataset.width, dataset.height, dataset.depth);
  printf("First 3 images labels: ");
  for (int i = 0; i < min(3, dataset.n); ++i) {
    printf("%d ", dataset.get_labels()[i]);
  }
  printf("\n\n");

  // Display sample pixel values from first 3 images to verify loading
  int image_size = dataset.width * dataset.height * dataset.depth;
  for (int img = 0; img < min(3, dataset.n); ++img) {
    printf("Image %d (label %d) - Sample pixel RGB values:\n", img, dataset.get_labels()[img]);
    float *image_data = dataset.get_data() + img * image_size;
    int plane_size = dataset.width * dataset.height;
    
    // Show first 5 pixels
    for (int pixel = 0; pixel < min(5, dataset.width * dataset.height); ++pixel) {
      float r = image_data[pixel];
      float g = image_data[pixel + plane_size];
      float b = image_data[pixel + 2 * plane_size];
      printf("  Pixel %d: R=%.3f, G=%.3f, B=%.3f\n", pixel, r, g, b);
    }
    printf("\n");
  }
  printf("\n");

  puts("=======================TRAINING GPU AUTOENCODER=======================");
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
  buffer.write(reinterpret_cast<char *>(encoded_dataset.get_data()),
               encoded_dataset_size);
  buffer.close();

  buffer.open(ENCODED_LABEL_FILE, ios::out | ios::binary);
  buffer.write(reinterpret_cast<char *>(encoded_dataset.get_labels()),
               encoded_dataset.n * sizeof(int));
  buffer.close();
}