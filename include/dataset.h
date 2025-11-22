#ifndef DATA_H
#define DATA_H

// Use unique_ptr to prevent memory leak
#include <memory>
#include <vector>

using std::move;
using std::unique_ptr, std::make_unique;
using std::vector;

/**
 * @brief Struct that represents a list of image (either encoded or decoded)
 *
 */
struct Dataset {
  // The list of images (flattened)
  unique_ptr<float[]> data;
  // The list of the labels
  unique_ptr<int[]> labels;
  // The number of images in the list
  int n;
  // The width of the image
  int width;
  // The bit-depth of the image
  int depth;

  /**
   * @brief Create an unitialized dataset
   *
   * @param n The number of images
   * @param width The width of the images
   * @param depth The depth of the images
   */
  Dataset(int n, int width, int depth);

  /**
   * @brief Initializes a dataset without labels
   *
   * @param data The flattened images
   * @param n The number of images
   * @param width The width of the images
   * @param depth The depth of the images
   */
  Dataset(unique_ptr<float[]> &data, int n, int width, int depth);

  /**
   * @brief Initializes a full dataset
   *
   * @param data The flattened images
   * @param labels The list of labels for the corresponding image
   * @param n The number of images
   * @param width The width of the images
   * @param depth The depth of the images
   */
  Dataset(unique_ptr<float[]> &data,
          unique_ptr<int[]>   &labels,
          int                  n,
          int                  width,
          int                  depth);

  /**
   * @brief Get the images of the dataset
   *
   * @return float* The flatten images
   */
  float *get_data() const;
  /**
   * @brief Get the labels of the dataset
   *
   * @return int* The list of labels
   */
  int *get_labels() const;
};

/**
 * @brief Load dataset from a directory
 *
 * @param dataset_dir The path to the dataset
 * @return Dataset The read dataset
 */
Dataset load_dataset(const char *dataset_dir);

/**
 * @brief Shuffle the dataset
 *
 * @param dataset The dataset to be shuffle
 */
void shuffle_dataset(Dataset &dataset);

/**
 * @brief Create a mini batches from a dataset
 *
 * @param dataset The dataset
 * @param batch_size The mini batch size
 * @return vector<Dataset> The list of batches created
 */
vector<Dataset> create_minibatches(const Dataset &dataset, int batch_size);

#endif