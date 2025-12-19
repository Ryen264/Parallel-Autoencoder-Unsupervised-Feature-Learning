#include "optimized_data_loader.h"

Optimized_Dataset::Optimized_Dataset()
    : data(0), labels(0), n(0), width(0), height(0), depth(0) {};

Optimized_Dataset::Optimized_Dataset(int n, int width, int height, int depth)
    : n(n), width(width), height(height), depth(depth) {
  int dataset_size = n * width * height * depth;
  CUDA_CHECK(cudaMallocHost(&data, dataset_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&labels, n * sizeof(int)));
}

Optimized_Dataset::Optimized_Dataset(
    float *data, int n, int width, int height, int depth)
    : n(n), width(width), height(height), depth(depth) {
  int dataset_size = n * width * height * depth;
  CUDA_CHECK(cudaMallocHost(&data, dataset_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&labels, n * sizeof(int)));

  CUDA_CHECK(
      cudaMemcpy(this->data, data, dataset_size * sizeof(float), cudaMemcpyHostToHost));
}

Optimized_Dataset::Optimized_Dataset(
    float *data, int *labels, int n, int width, int height, int depth)
    : n(n), width(width), height(height), depth(depth) {
  int dataset_size = n * width * height * depth;
  CUDA_CHECK(cudaMallocHost(&data, dataset_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&labels, n * sizeof(int)));

  CUDA_CHECK(
      cudaMemcpy(this->data, data, dataset_size * sizeof(float), cudaMemcpyHostToHost));
  CUDA_CHECK(cudaMemcpy(this->labels, labels, n * sizeof(int), cudaMemcpyHostToHost));
}

Optimized_Dataset::Optimized_Dataset(const Optimized_Dataset &other)
    : Optimized_Dataset(
          other.data, other.labels, other.n, other.width, other.height, other.depth) {};

Optimized_Dataset::Optimized_Dataset(Optimized_Dataset &&other)
    : data(other.data)
    , labels(other.labels)
    , n(other.n)
    , width(other.width)
    , height(other.height)
    , depth(other.depth) {
  other.data   = 0;
  other.labels = 0;
}

Optimized_Dataset::~Optimized_Dataset() {
  if (data) {
    CUDA_CHECK(cudaFreeHost(data));
    data = 0;
  }

  if (labels) {
    CUDA_CHECK(cudaFreeHost(labels));
    labels = 0;
  }
}

Optimized_Dataset &Optimized_Dataset::operator=(const Optimized_Dataset &other) {
  this->~Optimized_Dataset();

  n      = other.n;
  width  = other.width;
  height = other.height;
  depth  = other.depth;

  int dataset_size = n * width * height * depth;
  CUDA_CHECK(cudaMallocHost(&data, dataset_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&labels, n * sizeof(int)));

  CUDA_CHECK(
      cudaMemcpy(data, other.data, dataset_size * sizeof(float), cudaMemcpyHostToHost));
  CUDA_CHECK(cudaMemcpy(labels, other.labels, n * sizeof(int), cudaMemcpyHostToHost));

  return *this;
}

Optimized_Dataset &Optimized_Dataset::operator=(Optimized_Dataset &&other) {
  this->~Optimized_Dataset();

  data   = other.data;
  labels = other.labels;
  n      = other.n;
  width  = other.width;
  height = other.height;
  depth  = other.depth;

  other.data   = 0;
  other.labels = 0;

  return *this;
}

// CUDA kernel for normalization: convert uint8 [0, 255] to float [0, 1]
__global__ void normalizeKernel(unsigned char *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = input[idx] / 255.0f;
}

// Read a single CIFAR-10 binary file
static void
readBinaryFile(const char *filepath, unsigned char **raw_data, int num_samples) {
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
    if (raw_data)
      free((void *)*raw_data);
    else {
      fprintf(stderr, "Error: Memory deallocation failed\n");
    }
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fclose(file);
}

// Parse raw data and normalize using CUDA
static void parseAndNormalize(unsigned char *raw_data,
                              float         *images,
                              int           *labels,
                              int            num_samples,
                              bool           use_cuda) {
  int record_size = 1 + IMAGE_SIZE;

  // Extract labels
  for (int i = 0; i < num_samples; i++)
    labels[i] = (int)raw_data[i * record_size];

  if (use_cuda) {
    // Allocate device memory for raw image data
    unsigned char *d_raw_images;
    float         *d_images;
    int            image_data_size = num_samples * IMAGE_SIZE;

    CUDA_CHECK(cudaMalloc(&d_raw_images, image_data_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_images, image_data_size * sizeof(float)));

    // Copy raw image data to device (skip labels)
    unsigned char *raw_images = (unsigned char *)malloc(image_data_size);
    for (int i = 0; i < num_samples; i++)
      memcpy(raw_images + i * IMAGE_SIZE, raw_data + i * record_size + 1, IMAGE_SIZE);

    CUDA_CHECK(cudaMemcpy(d_raw_images,
                          raw_images,
                          image_data_size * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Launch normalization kernel
    int blocksPerGrid = (image_data_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

    normalizeKernel<<<blocksPerGrid, MAX_BLOCK_SIZE>>>(
        d_raw_images, d_images, image_data_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy normalized data back to host
    CUDA_CHECK(cudaMemcpy(
        images, d_images, image_data_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    if (raw_images)
      free((void *)raw_images);
    else {
      fprintf(stderr, "Error: Memory deallocation failed\n");
    }
    CUDA_CHECK(cudaFree((void *)d_raw_images));
    CUDA_CHECK(cudaFree((void *)d_images));
  } else {
    // CPU normalization
    for (int i = 0; i < num_samples; i++)
      for (int j = 0; j < IMAGE_SIZE; j++)
        images[i * IMAGE_SIZE + j] = raw_data[i * record_size + 1 + j] / 255.0f;
  }
}

Optimized_Dataset load_dataset(const char *dataset_dir,
                               int         n_batches,
                               bool        is_train) { // Check if CUDA is available
  int  deviceCount;
  bool use_cuda = (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0);

  int num_samples = is_train ? n_batches * NUM_PER_BATCH : NUM_TEST_SAMPLES;

  Optimized_Dataset dataset(num_samples, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_DEPTH);

  if (!images || !labels) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  if (is_train) {
    printf("Loading training data from %s...\n", dataset_dir);
    for (int batch = 1; batch <= n_batches; batch++) {
      char filepath[512];
      snprintf(filepath, sizeof(filepath), "%s/data_batch_%d.bin", dataset_dir, batch);

      unsigned char *raw_data;
      readBinaryFile(filepath, &raw_data, NUM_TRAIN_SAMPLES / NUM_BATCHES);

      int offset = (batch - 1) * (NUM_TRAIN_SAMPLES / NUM_BATCHES);
      parseAndNormalize(raw_data,
                        dataset.data + offset * IMAGE_SIZE,
                        dataset.labels + offset,
                        NUM_TRAIN_SAMPLES / NUM_BATCHES,
                        use_cuda);

      if (raw_data)
        free((void *)raw_data);
      else {
        fprintf(stderr, "Error: Memory deallocation failed\n");
      }
      printf("  ✓ Loaded batch %d/%d\n", batch, n_batches);
    }
    printf("✓ Training data loaded: %d samples\n", num_samples);
  } else {
    // Load test data
    printf("Loading test data from %s...\n", dataset_dir);
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/test_batch.bin", dataset_dir);

    unsigned char *raw_data;
    readBinaryFile(filepath, &raw_data, NUM_TEST_SAMPLES);
    parseAndNormalize(
        raw_data, dataset.data, dataset.labels, NUM_TEST_SAMPLES, use_cuda);
    if (raw_data)
      free((void *)raw_data);
    else {
      fprintf(stderr, "Error: Memory deallocation failed\n");
    }
    printf("✓ Test data loaded: %d samples\n", num_samples);
  }

  // Verify normalization
  float min_val = dataset.data[0], max_val = dataset.data[0];
  for (int i = 0; i < num_samples * IMAGE_SIZE; i++) {
    if (dataset.data[i] < min_val)
      min_val = dataset.data[i];
    if (dataset.data[i] > max_val)
      max_val = dataset.data[i];
  }
  printf("  Data range: [%.4f, %.4f]\n", min_val, max_val);

  return dataset;
}

void shuffle_dataset(Dataset &dataset) {
  int    n           = dataset.n;
  int    image_size  = dataset.width * dataset.height * dataset.depth;
  int    image_bytes = image_size * sizeof(float);
  float *data        = dataset.data;
  int   *labels      = dataset.labels;

  float *new_data;
  int   *new_labels;

  CUDA_CHECK(cudaMallocHost(&new_data, n * image_bytes));
  CUDA_CHECK(cudaMallocHost(&new_labels, n * sizeof(int)));

  // Create list of indices
  vector<int> indices(n);
  for (int i = 0; i < n; ++i)
    indices[i] = i;

  // Shuffle the indices
  mt19937 rng(time(nullptr));
  shuffle(indices.begin(), indices.end(), rng);

  // Copy data base on indices
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaMemcpy(new_data + i * image_size,
                          data + indices[i] * image_size,
                          image_bytes,
                          cudaMempcyHostToHost));
    CUDA_CHECK(cudaMemcpy(
        new_labels + i, labels + indices[i], sizeof(int), cudaMempcyHostToHost));
  }

  // Change the pointers of the original dataset
  dataset.data   = new_data;
  dataset.labels = new_labels;

  CUDA_CHECK(cudaFreeHost(data));
  CUDA_CHECK(cudaFreeHost(labels));
}