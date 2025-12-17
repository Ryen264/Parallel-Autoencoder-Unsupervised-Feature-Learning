#ifndef DATASET_H
#define DATASET_H

#include "constants.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <cuda_runtime.h>
using namespace std;

/**
 * @brief Struct that represents a list of image (either encoded or decoded)
 *
 */
struct Dataset {
    unique_ptr<float[]> data;   // The list of images (flattened)
    unique_ptr<int[]> labels;   // The list of the labels
    int n;                      // The number of images in the list
    int width;                  // The width of the image
    int height;                 // The height of the image
    int depth;                  // The depth of the image

    /**
     * @brief Default constructor - creates an empty dataset
     */
    Dataset();

    /**
     * @brief Create an unitialized dataset
     *
     * @param n The number of images
     * @param width The width of the images
     * @param height The height of the images
     * @param depth The depth of the images
     */
    Dataset(int n, int width, int height, int depth);

    /**
     * @brief Initializes a dataset without labels
     *
     * @param data The flattened images
     * @param n The number of images
     * @param width The width of the images
     * @param height The height of the images
     * @param depth The depth of the images
     */
    Dataset(unique_ptr<float[]> &data, int n, int width, int height, int depth);

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
    Dataset(unique_ptr<float[]> &data, unique_ptr<int[]> &labels, int n, int width, int height, int depth);

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
 * @brief Load CIFAR-10 dataset from binary files with options
 *
 * @param dataset_dir The path to the CIFAR-10 dataset directory
 * @param is_train True for training set, false for test set
 * @return Dataset The loaded and normalized dataset
 */
Dataset load_dataset(const char *dataset_dir, bool is_train = true);

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