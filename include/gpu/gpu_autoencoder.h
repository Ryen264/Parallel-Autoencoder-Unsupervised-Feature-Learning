#ifndef CPU_AUTOENCODER_H
#define CPU_AUTOENCODER_H

#include "data_loader.h"
#include "gpu_unique_ptr.h"

/**
 * @brief The class that encapsulates the network using CPU
 *
 */
class Gpu_Autoencoder {
  Gpu_Unique_Ptr _encoder_filter_1;
  Gpu_Unique_Ptr _encoder_bias_1;

  // Save output for backwards propogation
  Gpu_Unique_Ptr _out_encoder_filter_1;
  Gpu_Unique_Ptr _out_encoder_bias_1;

  Gpu_Unique_Ptr _out_encoder_relu_1;
  Gpu_Unique_Ptr _out_max_pooling_1;

  Gpu_Unique_Ptr _encoder_filter_2;
  Gpu_Unique_Ptr _encoder_bias_2;
  Gpu_Unique_Ptr _out_encoder_filter_2;
  Gpu_Unique_Ptr _out_encoder_bias_2;

  Gpu_Unique_Ptr _out_encoder_relu_2;
  Gpu_Unique_Ptr _out_max_pooling_2;

  Gpu_Unique_Ptr _decoder_filter_1;
  Gpu_Unique_Ptr _decoder_bias_1;
  Gpu_Unique_Ptr _out_decoder_filter_1;
  Gpu_Unique_Ptr _out_decoder_bias_1;

  Gpu_Unique_Ptr _out_decoder_relu_1;
  Gpu_Unique_Ptr _out_upsampling_1;

  Gpu_Unique_Ptr _decoder_filter_2;
  Gpu_Unique_Ptr _decoder_bias_2;
  Gpu_Unique_Ptr _out_decoder_filter_2;
  Gpu_Unique_Ptr _out_decoder_bias_2;

  Gpu_Unique_Ptr _out_decoder_relu_2;
  Gpu_Unique_Ptr _out_upsampling_2;

  Gpu_Unique_Ptr _decoder_filter_3;
  Gpu_Unique_Ptr _decoder_bias_3;
  Gpu_Unique_Ptr _out_decoder_filter_3;
  Gpu_Unique_Ptr _out_decoder_bias_3;

  // Gradients
  Gpu_Unique_Ptr _d_in;
  Gpu_Unique_Ptr _d_out;
  Gpu_Unique_Ptr _d_filter;

  // Batch data
  Gpu_Unique_Ptr _batch_data;
  Gpu_Unique_Ptr _res_data;

  // Block sizes
  static constexpr dim3 _block_size_1D   = dim3(1024);
  static constexpr dim3 _block_size_3D_1 = dim3(32, 32, 1);
  static constexpr dim3 _block_size_3D_2 = dim3(16, 16, 4);
  static constexpr dim3 _block_size_3D_3 = dim3(8, 8, 16);

  /**
   * @brief Perform a formward pass
   *
   * @param dataset The dataset to encode
   */
  void _forward_pass(const Dataset &dataset);

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
  float _fit_batch(const Dataset &batch, float learning_rate);

public:
  /**
   * @brief Initialize a new autoencoder with random paramters
   *
   */
  Gpu_Autoencoder();

  /**
   * @brief Reads the parameters from a file
   *
   * @param filename The file containing the model's parameters
   */
  Gpu_Autoencoder(const char *filename);

  /**
   * @brief Destroy the Autoencoder object
   *
   */
  ~Gpu_Autoencoder() = default;

  /**
   * @brief Encodes a dataset
   *
   * @param dataset The dataset to be encoded
   * @return Dataset The encoded dataset
   */
  Dataset encode(const Dataset &dataset) const;

  /**
   * @brief Decodes a dataset
   *
   * @param dataset The dataset to be decoded
   * @return Dataset The decoded dataset
   */
  Dataset decode(const Dataset &dataset) const;

  /**
   * @brief Trains the model using a dataset
   *
   * @param dataset The dataset
   * @param n_epoch The number of epochs
   * @param batch_size Minibatch size (set to 0 to disable)
   * @param learning_rate Learning rate of the model
   * @param verbose Whether to disable more or less information
   * @param checkpoint Save the model's parameter after a specific number of epochs (set
   * to 0 to disable)
   * @param output_dir The file to save model's param
   */
  void fit(const Dataset &dataset,
           int            n_epoch,
           int            batch_size,
           float          learning_rate,
           bool           verbose,
           int            checkpoint,
           const char    *output_dir = "./model");

  /**
   * @brief Evaluate the model
   *
   * @param dataset The dataset to be evaluated
   * @return float The MSE between the actual and expected result
   */
  float eval(const Dataset &dataset) const;

  /**
   * @brief Write the model's parameters to a file
   *
   * @param filename The file to write the model's parameter
   */
  void save_parameters(const char *filename) const;
};

#endif