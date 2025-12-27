#ifndef OPT2_AUTOENCODER_H
#define OPT2_AUTOENCODER_H

#include "optimized2_layers.h"
#include "optimized_data_loader.h"
#include "progress_bar.h"
#include "timer.h"
#include "utils.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
using namespace std;

/**
 * @brief The class that encapsulates the network using CPU
 *
 */
class Optimized2_Autoencoder {
  float *_encoder_filter_1;
  float *_encoder_bias_1;

  // Save output for backwards propogation
  float *_out_encoder_filter_1;
  float *_in_encoder_relu_1;

  float *_out_max_pooling_1;

  float *_encoder_filter_2;
  float *_encoder_bias_2;

  float *_out_encoder_filter_2;
  float *_in_encoder_relu_2;

  float *_out_max_pooling_2;

  float *_decoder_filter_1;
  float *_decoder_bias_1;

  float *_out_decoder_filter_1;
  float *_in_decoder_relu_1;

  float *_out_upsampling_1;

  float *_decoder_filter_2;
  float *_decoder_bias_2;

  float *_out_decoder_filter_2;
  float *_in_decoder_relu_2;

  float *_out_upsampling_2;

  float *_decoder_filter_3;
  float *_decoder_bias_3;

  float *_out_decoder_filter_3;
  float *_in_decoder_relu_3;

  // Gradients
  float *_d_in;
  float *_d_out;
  float *_d_filter;
  float *_d_bias;

  // Batch data
  float *_batch_data;
  float *_res_data;

  // Block sizes
  static constexpr dim3 _block_size_1D     = dim3(1024);
  static constexpr dim3 _block_size_3D_1   = dim3(32, 32, 1);
  static constexpr dim3 _block_size_3D_2   = dim3(16, 16, 4);
  static constexpr dim3 _block_size_3D_3   = dim3(8, 8, 16);

  /**
   * @brief Perform a formward pass
   *
   */
  void _forward_pass(float *data, int n, int width, int height, int depth);

  /**
   * @brief Allocate memory for the parameters
   *
   */
  void _allocate_mem();

  /**
   * @brief Allocate memory for the output arrays
   *
   * @param n The number of images
   * @param width The width of the images
   * @param height The height of the images
   *
   */
  void _allocate_output_mem(int n, int width, int height);

  /**
   * @brief Deallocate memory by the output arrays
   *
   */
  void _deallocate_output_mem();

  /**
   * @brief Fit a minibatch to the model
   *
   * @param batch The batch to fit
   * @param learning_rate The learning rate
   * @return float Loss of the minibatch
   */
  float _fit_batch(float *data, int n, int width, int height, int depth, float learning_rate);

public:
  /**
   * @brief Initialize a new autoencoder with random paramters
   *
   */
  Optimized2_Autoencoder();

  /**
   * @brief Reads the parameters from a file
   *
   * @param filename The file containing the model's parameters
   */
  Optimized2_Autoencoder(const char *filename);

  /**
   * @brief Destroy the Autoencoder object
   *
   */
  ~Optimized2_Autoencoder();

  /**
   * @brief Encodes a dataset
   *
   * @param dataset The dataset to be encoded
   * @return Optimized_Dataset The encoded dataset
   */
  Optimized_Dataset encode(const Optimized_Dataset &dataset) const;

  /**
   * @brief Decodes a dataset
   *
   * @param dataset The dataset to be decoded
   * @return Optimized_Dataset The decoded dataset
   */
  Optimized_Dataset decode(const Optimized_Dataset &dataset) const;

  /**
   * @brief Trains the model using a dataset
   *
   * @param dataset The dataset
   * @param n_epoch The number of epochs
   * @param batch_size Minibatch size (set to 0 to disable)
   * @param learning_rate Learning rate of the model
   * @param checkpoint Save the model's parameter after a specific number of epochs (set
   * to 0 to disable)
   * @param output_dir The file to save model's param
   */
  void fit(const Optimized_Dataset &dataset,
           int                      n_epoch,
           int                      batch_size,
           float                    learning_rate,
           int                      checkpoint,
           const char              *output_dir = "./model");

  /**
   * @brief Evaluate the model
   *
   * @param dataset The dataset to be evaluated
   * @return float The MSE between the actual and expected result
   */
  float eval(const Optimized_Dataset &dataset) const;

  /**
   * @brief Write the model's parameters to a file
   *
   * @param filename The file to write the model's parameter
   */
  void save_parameters(const char *filename) const;

  /**
   * @brief Load the model's parameters from a file
   *
   * @param filename The file containing the model's parameters
   */
  void load_parameters(const char *filename);

};

#endif