#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "constants.h"
#include "data_loader.h"

/**
 * @brief Interface class that encapsulates the network
 *
 */
class IAutoencoder {
protected:
  // The first conv2d layer of the encoder
  static constexpr int _ENCODER_FILTER_1_DEPTH = 256;
  static constexpr int _ENCODER_FILTER_1_SIZE =
      CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * IMAGE_DEPTH * _ENCODER_FILTER_1_DEPTH;

  // The second conv2d layer of the encoder
  static constexpr int _ENCODER_FILTER_2_DEPTH = 128;
  static constexpr int _ENCODER_FILTER_2_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_HEIGHT *
                                                _ENCODER_FILTER_1_DEPTH *
                                                _ENCODER_FILTER_2_DEPTH;

  // The first conv2d layer of the decoder
  static constexpr int _DECODER_FILTER_1_DEPTH = 128;
  static constexpr int _DECODER_FILTER_1_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_HEIGHT *
                                                _ENCODER_FILTER_2_DEPTH *
                                                _DECODER_FILTER_1_DEPTH;

  // The second conv2d layer of the decoder
  static constexpr int _DECODER_FILTER_2_DEPTH = 256;
  static constexpr int _DECODER_FILTER_2_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_HEIGHT *
                                                _DECODER_FILTER_1_DEPTH *
                                                _DECODER_FILTER_2_DEPTH;

  // The third conv2d layer of the decoder
  static constexpr int _DECODER_FILTER_3_DEPTH = IMAGE_DEPTH;
  static constexpr int _DECODER_FILTER_3_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_HEIGHT *
                                                _DECODER_FILTER_2_DEPTH *
                                                _DECODER_FILTER_3_DEPTH;

  // Encode and decode by batches to use less memory
  static constexpr int _ENCODE_BATCH_SIZE = 1024;

public:
  /**
   * @brief Destroy the Autoencoder object
   *
   */
  virtual ~IAutoencoder() = default;

  /**
   * @brief Encodes a dataset
   *
   * @param dataset The dataset to be encoded
   * @return Dataset The encoded dataset
   */
  virtual Dataset encode(const Dataset &dataset) const = 0;

  /**
   * @brief Decodes a dataset
   *
   * @param dataset The dataset to be decoded
   * @return Dataset The decoded dataset
   */
  virtual Dataset decode(const Dataset &dataset) const = 0;

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
  virtual void fit(const Dataset &dataset,
                   int            n_epoch,
                   int            batch_size,
                   float          learning_rate,
                   bool           verbose,
                   int            checkpoint,
                   const char    *output_dir = "./model") = 0;

  /**
   * @brief Evaluate the model
   *
   * @param dataset The dataset to be evaluated
   * @return float The MSE between the actual and expected result
   */
  virtual float eval(const Dataset &dataset) = 0;

  /**
   * @brief Write the model's parameters to a file
   *
   * @param filename The file to write the model's parameter
   */
  virtual void save_paramters(const char *filename) const = 0;
};

#endif
