#include "cpu_data_loader.h"

// Thay thế parseAndNormalize bằng triển khai CPU thuần túy
static void parseAndNormalize(unsigned char *raw_data,
                              float         *images,
                              int           *labels,
                              int            num_samples) {
  int record_size = 1 + IMAGE_SIZE;

  // Extract labels and CPU normalization
  for (int i = 0; i < num_samples; i++) {
    labels[i] = (int)raw_data[i * record_size];
    for (int j = 0; j < IMAGE_SIZE; j++)
      // Normalize: convert uint8 [0, 255] to float [0, 1]
      images[i * IMAGE_SIZE + j] = raw_data[i * record_size + 1 + j] / 255.0f;
  }
}

// Read a single CIFAR-10 binary file
static void readBinaryFile(const char *filepath, unsigned char **raw_data, int num_samples) {
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s\n", filepath);
    exit(EXIT_FAILURE);
  }

  int record_size = 1 + IMAGE_SIZE; // 1 byte label + 3072 bytes image
  int total_size  = num_samples * record_size;

  *raw_data = (unsigned char *)malloc(total_size);
  if (!*raw_data) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  size_t read_size = fread(*raw_data, 1, total_size, file);
  if (read_size != total_size) {
    fprintf(stderr, "Error: Could not read complete file %s\n", filepath);
    if (*raw_data)
      free(*raw_data);
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fclose(file);
}