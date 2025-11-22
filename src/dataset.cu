#include "dataset.h"
#include <algorithm>
#include <cstring>

using std::memcpy;
using std::random_shuffle;

Dataset::Dataset(int n, int width, int depth)
    : data(make_unique<float[]>(n * width * width * depth))
    , labels(make_unique<int[]>(n))
    , n(n)
    , width(width)
    , depth(depth) {};

Dataset::Dataset(unique_ptr<float[]> &data, int n, int width, int depth)
    : data(move(data))
    , labels(make_unique<int[]>(n))
    , n(n)
    , width(width)
    , depth(depth) {};

Dataset::Dataset(
    unique_ptr<float[]> &data, unique_ptr<int[]> &labels, int n, int width, int depth)
    : data(move(data)), labels(move(labels)), n(n), width(width), depth(depth) {};

float *Dataset::get_data() const { return data.get(); }
int   *Dataset::get_labels() const { return labels.get(); }

Dataset load_dataset(const char *dataset_dir) {
  // TODO
}

void shuffle_dataset(Dataset &dataset) {
  int                 n           = dataset.n;
  int                 image_size  = dataset.width * dataset.width * dataset.depth;
  int                 image_bytes = image_size * sizeof(float);
  float              *data        = dataset.get_data();
  int                *labels      = dataset.get_labels();
  unique_ptr<float[]> new_data    = make_unique<float[]>(n * image_size);
  unique_ptr<int[]>   new_labels  = make_unique<int[]>(n);

  // Create list of indices
  vector<int> indices(n);
  for (int i = 0; i < n; ++i)
    indices[i] = i;

  // Shuffle the indices
  random_shuffle(indices.begin(), indices.end());

  // Copy data base on indices
  for (int i = 0; i < n; ++i) {
    memcpy(
        new_data.get() + i * image_bytes, data + indices[i] * image_bytes, image_bytes);
    memcpy(new_labels.get() + i * sizeof(int),
           labels + indices[i] * sizeof(int),
           sizeof(int));
  }

  // Change the pointers of the original dataset
  dataset.data   = move(new_data);
  dataset.labels = move(new_labels);
}

vector<Dataset> create_minibatches(const Dataset &dataset, int batch_size) {
  int    n_batches  = (dataset.n + batch_size - 1) / batch_size;
  float *data       = dataset.get_data();
  int   *labels     = dataset.get_labels();
  int    image_size = dataset.width * dataset.width * dataset.depth * sizeof(float);
  int    batch_image_bytes  = batch_size * image_size;
  int    batch_labels_bytes = batch_size * sizeof(int);
  vector<Dataset> batches(n_batches, Dataset(batch_size, dataset.width, dataset.depth));

  // Copy data
  for (int i = 0; i < n_batches; ++i) {
    memcpy(batches[i].get_data(), data + i * batch_image_bytes, batch_image_bytes);
    memcpy(
        batches[i].get_labels(), labels + i * batch_labels_bytes, batch_labels_bytes);
  }

  // Create last batch
  int     n_remaining = dataset.n - batch_size * n_batches;
  Dataset last_batch(n_remaining, dataset.width, dataset.depth);
  memcpy(last_batch.get_data(),
         data + n_batches * batch_image_bytes,
         n_remaining * image_size);
  memcpy(last_batch.get_labels(),
         labels + n_batches * batch_labels_bytes,
         n_remaining * sizeof(int));

  // Push into vector
  batches.emplace_back(
      last_batch.data, last_batch.labels, n_remaining, dataset.width, dataset.depth);

  return batches;
}