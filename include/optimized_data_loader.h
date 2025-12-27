#ifndef OPT_DATASET_H
#define OPT_DATASET_H

#include "constants.h"
#include "macro.h"

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
using namespace std;

/**
 * @brief Struct that represents a list of image (either encoded or decoded)
 *
 */
struct Optimized_Dataset {
  // The list of images (flattened)
  float *data;
  // The list of the labels
  int *labels;
  // The number of images in the list
  int n;
  // The width of the image
  int width;
  // The height of the image
  int height;
  // The bit-depth of the image
  int depth;

  /**
   * @brief Default constructor - creates an empty dataset
   */
  Optimized_Dataset();

  /**
   * @brief Create an unitialized dataset
   *
   * @param n The number of images
   * @param width The width of the images
   * @param height The height of the images
   * @param depth The depth of the images
   */
  Optimized_Dataset(int n, int width, int height, int depth);

  /**
   * @brief Initializes a dataset without labels
   *
   * @param data The flattened images
   * @param n The number of images
   * @param width The width of the images
   * @param height The height of the images
   * @param depth The depth of the images
   */
  Optimized_Dataset(float *data, int n, int width, int height, int depth);

  /**
   * @brief Initializes a full dataset
   *
   * @param data The flattened images
   * @param labels The list of labels for the corresponding image
   * @param n The number of images
   * @param width The width of the images
   * @param height The height of the images
   * @param depth The depth of the images
   */
  Optimized_Dataset(float *data, int *labels, int n, int width, int height, int depth);
  Optimized_Dataset(const Optimized_Dataset &other);
  Optimized_Dataset(Optimized_Dataset &&other);

  ~Optimized_Dataset();

  Optimized_Dataset &operator=(const Optimized_Dataset &other);
  Optimized_Dataset &operator=(Optimized_Dataset &&other);
};

/**
 * @brief Read CIFAR-10 dataset from binary files with options
 *
 * @param dataset_dir The path to the CIFAR-10 dataset directory
 * @param n_batches The number of batches to read
 * @param is_train True for training set, false for test set
 * @return Dataset The loaded and normalized dataset
 */
Optimized_Dataset read_optimized_dataset(const char *dataset_dir, int n_batches, bool is_train = true);

/**
 * @brief Shuffle the dataset
 *
 * @param dataset The dataset to be shuffle
 */
void shuffle_optimized_dataset(Optimized_Dataset &dataset);

#endif