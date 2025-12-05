#ifndef CPU_AUTOENCODER_H
#define CPU_AUTOENCODER_H

#include "data_loader.h"

/**
 * @brief The class that encapsulates the network using CPU
 *
 */
class Cpu_Autoencoder {

  unique_ptr<float[]> _encoder_filter_1;
  unique_ptr<float[]> _encoder_bias_1;

  // Save output for backwards propogation
  unique_ptr<float[]> _out_encoder_filter_1;
  unique_ptr<float[]> _out_encoder_bias_1;

  unique_ptr<float[]> _out_encoder_relu_1;
  unique_ptr<float[]> _out_max_pooling_1;

  unique_ptr<float[]> _encoder_filter_2;
  unique_ptr<float[]> _encoder_bias_2;
  unique_ptr<float[]> _out_encoder_filter_2;
  unique_ptr<float[]> _out_encoder_bias_2;

  unique_ptr<float[]> _out_encoder_relu_2;
  unique_ptr<float[]> _out_max_pooling_2;

  unique_ptr<float[]> _decoder_filter_1;
  unique_ptr<float[]> _decoder_bias_1;
  unique_ptr<float[]> _out_decoder_filter_1;
  unique_ptr<float[]> _out_decoder_bias_1;

  unique_ptr<float[]> _out_decoder_relu_1;
  unique_ptr<float[]> _out_upsampling_1;

  unique_ptr<float[]> _decoder_filter_2;
  unique_ptr<float[]> _decoder_bias_2;
  unique_ptr<float[]> _out_decoder_filter_2;
  unique_ptr<float[]> _out_decoder_bias_2;

  unique_ptr<float[]> _out_decoder_relu_2;
  unique_ptr<float[]> _out_upsampling_2;

  unique_ptr<float[]> _decoder_filter_3;
  unique_ptr<float[]> _decoder_bias_3;
  unique_ptr<float[]> _out_decoder_filter_3;
  unique_ptr<float[]> _out_decoder_bias_3;

  // Gradients
  unique_ptr<float[]> _d_in;
  unique_ptr<float[]> _d_out;
  unique_ptr<float[]> _d_filter;

  /**
   * @brief Encode while saving outputs for training
   *
   * @param dataset The dataset to encode
   * @return Dataset The encoded dataset
   */
  Dataset _encode_save_output(const Dataset &dataset);

  /**
   * @brief Decode while saving outputs for training
   *
   * @param dataset The dataset to decode
   * @return Dataset The decoded dataset
   */
  Dataset _decode_save_output(const Dataset &dataset);

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
  Cpu_Autoencoder();

  /**
   * @brief Reads the parameters from a file
   *
   * @param filename The file containing the model's parameters
   */
  Cpu_Autoencoder(const char *filename);

  /**
   * @brief Destroy the Autoencoder object
   *
   */
  ~Cpu_Autoencoder() = default;

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
  float eval(const Dataset &dataset);

  /**
   * @brief Write the model's parameters to a file
   *
   * @param filename The file to write the model's parameter
   */
  void save_paramters(const char *filename) const;
};

#endif