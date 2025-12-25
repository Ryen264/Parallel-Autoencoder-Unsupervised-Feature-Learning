#include "data_loader.h"
#include <iostream>

// Include direct.h for _mkdir on Windows/MinGW
#ifdef _WIN32
#include <direct.h>
#endif

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
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fclose(file);
}

// Parse raw data and normalize using CPU
static void parseAndNormalize(unsigned char *raw_data,
                              float         *images,
                              int           *labels,
                              int            num_samples) {
  int record_size = 1 + IMAGE_SIZE;

  // Extract labels and normalize images on CPU
  for (int i = 0; i < num_samples; i++) {
    labels[i] = (int)raw_data[i * record_size];
    for (int j = 0; j < IMAGE_SIZE; j++)
      images[i * IMAGE_SIZE + j] = raw_data[i * record_size + 1 + j] / 255.0f;
  }
}

Dataset::Dataset()
    : data(nullptr), labels(nullptr), n(0), width(0), height(0), depth(0) {};

Dataset::Dataset(const Dataset &other)
    : n(other.n), width(other.width), height(other.height), depth(other.depth) {
  if (other.data) {
    int size = n * width * height * depth;
    data = make_unique<float[]>(size);
    memcpy(data.get(), other.data.get(), size * sizeof(float));
  } else {
    data = nullptr;
  }

  if (other.labels) {
    labels = make_unique<int[]>(n);
    memcpy(labels.get(), other.labels.get(), n * sizeof(int));
  } else {
    labels = nullptr;
  }
}

Dataset::Dataset(Dataset &&other) noexcept
    : data(std::move(other.data)), labels(std::move(other.labels)), n(other.n), width(other.width),
      height(other.height), depth(other.depth) {};

Dataset::Dataset(int n, int width, int height, int depth)
    : data(make_unique<float[]>(n * width * height * depth))
    , labels(make_unique<int[]>(n))
    , n(n)
    , width(width)
    , height(height)
    , depth(depth) {};

Dataset::Dataset(unique_ptr<float[]> &data, int n, int width, int height, int depth)
    : data(std::move(data))
    , labels(make_unique<int[]>(n))
    , n(n)
    , width(width)
    , height(height)
    , depth(depth) {};

Dataset::Dataset(unique_ptr<float[]> &data,
                 unique_ptr<int[]>   &labels,
                 int                  n,
                 int                  width,
                 int                  height,
                 int                  depth)
    : data(std::move(data))
    , labels(std::move(labels))
    , n(n)
    , width(width)
    , height(height)
    , depth(depth) {};

float *Dataset::get_data() const { return data.get(); }

int *Dataset::get_labels() const { return labels.get(); }

Dataset &Dataset::operator=(const Dataset &other) {
  if (this != &other) {
    // Copy data
    if (other.data) {
      int size = other.n * other.width * other.height * other.depth;
      data = make_unique<float[]>(size);
      memcpy(data.get(), other.data.get(), size * sizeof(float));
    } else {
      data = nullptr;
    }

    // Copy labels
    if (other.labels) {
      labels = make_unique<int[]>(other.n);
      memcpy(labels.get(), other.labels.get(), other.n * sizeof(int));
    } else {
      labels = nullptr;
    }

    n      = other.n;
    width  = other.width;
    height = other.height;
    depth  = other.depth;
  }
  return *this;
}

Dataset &Dataset::operator=(Dataset &&other) noexcept {
  if (this != &other) {
    data   = std::move(other.data);
    labels = std::move(other.labels);
    n      = other.n;
    width  = other.width;
    height = other.height;
    depth  = other.depth;
  }
  return *this;
}

// Read CIFAR-10 dataset from binary files
Dataset read_dataset(const char *dataset_dir, int n_batches, bool is_train) {
  int num_samples = is_train ? n_batches * NUM_PER_BATCH : NUM_TEST_SAMPLES;

  // Allocate memory for images and labels
  float *images = new float[num_samples * IMAGE_SIZE];
  int   *labels = new int[num_samples];

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
      
      // Fixed call: Removed undefined use_cuda argument
      parseAndNormalize(raw_data,
                        images + offset * IMAGE_SIZE,
                        labels + offset,
                        NUM_TRAIN_SAMPLES / NUM_BATCHES);

      if (raw_data) {
        free((void *)raw_data);
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
    
    // Fixed call: Removed undefined use_cuda argument
    parseAndNormalize(raw_data, images, labels, NUM_TEST_SAMPLES);
    
    if (raw_data) {
      free((void *)raw_data);
    }
    printf("✓ Test data loaded: %d samples\n", num_samples);
  }

  // Verify normalization
  float min_val = images[0], max_val = images[0];
  for (int i = 0; i < num_samples * IMAGE_SIZE; i++) {
    if (images[i] < min_val)
      min_val = images[i];
    if (images[i] > max_val)
      max_val = images[i];
  }
  printf("  Data range: [%.4f, %.4f]\n", min_val, max_val);

  // Convert to Dataset object
  unique_ptr<float[]> data_ptr(images);
  unique_ptr<int[]>   labels_ptr(labels);
  
  return Dataset(data_ptr, labels_ptr,
                num_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
}

void shuffle_dataset(Dataset &dataset) {
  int    n           = dataset.n;
  int    width       = dataset.width;
  int    height      = dataset.height;
  int    n_pixel     = width * height;
  int    depth       = dataset.depth;
  int    image_size  = n_pixel * depth;
  float *data        = dataset.get_data();
  int   *labels      = dataset.get_labels();

  unique_ptr<float[]> new_data   = make_unique<float[]>(n * image_size);
  unique_ptr<int[]>   new_labels = make_unique<int[]>(n);

  // Create list of indices
  vector<int> indices(n);
  for (int i = 0; i < n; ++i)
    indices[i] = i;

  // Shuffle the indices
  mt19937 rng(time(nullptr));
  shuffle(indices.begin(), indices.end(), rng);

  // Copy data base on indices
  for (int i = 0; i < n; ++i) {
    float *in_offset  = data + indices[i] * image_size;
    float *out_offset = new_data.get() + i * image_size;

    // Copy each color
    for (int c = 0; c < depth; ++c)
      memcpy(
          out_offset + c * n_pixel, in_offset + c * n_pixel, n_pixel * sizeof(float));
    memcpy(new_labels.get() + i, labels + indices[i], sizeof(int));
  }

  // Change the pointers of the original dataset
  dataset.data   = std::move(new_data);
  dataset.labels = std::move(new_labels);
}

vector<Dataset> create_minibatches(const Dataset &dataset, int batch_size) {
  int    n_batches  = (dataset.n + batch_size - 1) / batch_size;
  float *data       = dataset.get_data();
  int   *labels     = dataset.get_labels();
  int    image_size = dataset.width * dataset.height * dataset.depth;

  vector<Dataset> batches;
  batches.reserve(n_batches);

  // Create and copy data for each batch
  for (int i = 0; i < n_batches; ++i) {
    int current_batch_size =
        (i < n_batches - 1) ? batch_size : (dataset.n - i * batch_size);
    int current_batch_image_bytes  = current_batch_size * image_size * sizeof(float);
    int current_batch_labels_bytes = current_batch_size * sizeof(int);

    Dataset batch(current_batch_size, dataset.width, dataset.height, dataset.depth);

    memcpy(batch.get_data(),
           data + i * batch_size * image_size,
           current_batch_image_bytes);
    memcpy(batch.get_labels(), labels + i * batch_size, current_batch_labels_bytes);

    batches.push_back(std::move(batch));
  }
  return batches;
}

bool write_binary(const Dataset &dataset, const char *filepath) {
  // Check and create directory if needed
  size_t found = string(filepath).find_last_of("/\\");
  if (found != string::npos) {
    string dir_path = string(filepath).substr(0, found);
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
      printf("Creating directory: %s\n", dir_path.c_str());
      #ifdef _WIN32
        _mkdir(dir_path.c_str());
      #else
        mkdir(dir_path.c_str(), 0777);
      #endif
    }
  }

  FILE *file = fopen(filepath, "wb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s for writing\n", filepath);
    return false;
  }

  // Write metadata (dimensions)
  if (fwrite(&dataset.n, sizeof(int), 1, file) !=
      1 ||
      fwrite(&dataset.width, sizeof(int), 1, file) !=
      1 ||
      fwrite(&dataset.height, sizeof(int), 1, file) !=
      1 ||
      fwrite(&dataset.depth, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "Error: Failed to write metadata to %s\n", filepath);
    fclose(file);
    return false;
  }

  // Write image data
  int image_size    = dataset.width * dataset.height * dataset.depth;
  int data_elements = dataset.n * image_size;
  if (fwrite(dataset.get_data(), sizeof(float), data_elements, file) != data_elements) {
    fprintf(stderr, "Error: Failed to write image data to %s\n", filepath);
    fclose(file);
    return false;
  }

  // Write labels if available
  if (dataset.labels) {
    if (fwrite(dataset.get_labels(), sizeof(int), dataset.n, file) != dataset.n) {
      fprintf(stderr, "Error: Failed to write labels to %s\n", filepath);
      fclose(file);
      return false;
    }
  }

  fclose(file);
  printf("✓ Dataset saved to %s (%d samples, %dx%dx%d)\n",
         filepath,
         dataset.n,
         dataset.width,
         dataset.height,
         dataset.depth);
  return true;
}

Dataset read_binary(const char *filepath) {
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s for reading\n", filepath);
    exit(EXIT_FAILURE);
  }

  // Read metadata (dimensions)
  int n, width, height, depth;
  if (fread(&n, sizeof(int), 1, file) != 1 ||
      fread(&width, sizeof(int), 1, file) != 1 ||
      fread(&height, sizeof(int), 1, file) != 1 ||
      fread(&depth, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "Error: Failed to read metadata from %s\n", filepath);
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Allocate and read image data
  int image_size = width * height * depth;
  int data_elements = n * image_size;
  float *images = new float[data_elements];
  if (fread(images, sizeof(float), data_elements, file) != data_elements) {
    fprintf(stderr, "Error: Failed to read image data from %s\n", filepath);
    delete[] images;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Allocate and read labels
  int *labels = new int[n];
  size_t labels_read = fread(labels, sizeof(int), n, file);
  
  // Check if we reached EOF (labels might not be present)
  bool has_labels = (labels_read == n);
  if (!has_labels) {
    delete[] labels;
    labels = nullptr;
  }

  fclose(file);
  printf("✓ Dataset loaded from %s (%d samples, %dx%dx%d)\n",
         filepath, n, width, height, depth);

  // Convert to Dataset object
  unique_ptr<float[]> data_ptr(images);
  if (has_labels) {
    unique_ptr<int[]> labels_ptr(labels);
    return Dataset(data_ptr, labels_ptr, n, width, height, depth);
  } else {
    return Dataset(data_ptr, n, width, height, depth);
  }
}