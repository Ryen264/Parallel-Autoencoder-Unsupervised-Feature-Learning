#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "constants.h"
#include "images.h"

/**
 * @brief Interface class that encapsulates the network
 *
 */
class IAutoencoder {
protected:
  // The first conv2d layer of the encoder
  static constexpr int _ENCODER_FILTER_1_DEPTH = 256;
  static constexpr int _ENCODER_FILTER_1_SIZE =
      CONV_FILTER_WIDTH * CONV_FILTER_WIDTH * IMAGE_DEPTH * _ENCODER_FILTER_1_DEPTH;
  unique_ptr<float[]> _encoder_filter_1;
  unique_ptr<float[]> _encoder_bias_1;

  // The second conv2d layer of the encoder
  static constexpr int _ENCODER_FILTER_2_DEPTH = 128;
  static constexpr int _ENCODER_FILTER_2_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_WIDTH *
                                                _ENCODER_FILTER_1_DEPTH *
                                                _ENCODER_FILTER_2_DEPTH;
  unique_ptr<float[]> _encoder_filter_2;
  unique_ptr<float[]> _encoder_bias_2;

  // The first conv2d layer of the decoder
  static constexpr int _DECODER_FILTER_1_DEPTH = 128;
  static constexpr int _DECODER_FILTER_1_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_WIDTH *
                                                _ENCODER_FILTER_2_DEPTH *
                                                _DECODER_FILTER_1_DEPTH;
  unique_ptr<float[]> _decoder_filter_1;
  unique_ptr<float[]> _decoder_bias_1;

  // The second conv2d layer of the decoder
  static constexpr int _DECODER_FILTER_2_DEPTH = 256;
  static constexpr int _DECODER_FILTER_2_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_WIDTH *
                                                _DECODER_FILTER_1_DEPTH *
                                                _DECODER_FILTER_2_DEPTH;
  unique_ptr<float[]> _decoder_filter_2;
  unique_ptr<float[]> _decoder_bias_2;

  // The third conv2d layer of the decoder
  static constexpr int _DECODER_FILTER_3_DEPTH = IMAGE_DEPTH;
  static constexpr int _DECODER_FILTER_3_SIZE  = CONV_FILTER_WIDTH *
                                                CONV_FILTER_WIDTH *
                                                _DECODER_FILTER_2_DEPTH *
                                                _DECODER_FILTER_3_DEPTH;
  unique_ptr<float[]> _decoder_filter_3;
  unique_ptr<float[]> _decoder_bias_3;

  /**
   * @brief Allocate memory for the parameters
   *
   */
  void _allocate_mem();

  /**
   * @brief Write the model's parameters to a file
   *
   * @param filename The file to write the model's parameter
   */
  void _save_paramters(const char *filename) const;

public:
  /**
   * @brief Initialize a new autoencoder with random paramters
   *
   */
  IAutoencoder();
  /**
   * @brief Reads the parameters from a file
   *
   * @param filename The file containing the model's parameters
   */
  IAutoencoder(const char *filename);
  /**
   * @brief Destroy the Autoencoder object
   *
   */
  virtual ~IAutoencoder() = default;

  /**
   * @brief Encodes a list of images
   *
   * @param images The images to be encoded
   * @return Images The encoded images
   */
  virtual Images encode(const Images &images) const = 0;
  /**
   * @brief Decodes a list of images
   *
   * @param images The images to be decoded
   * @return Images The decoded images
   */
  virtual Images decode(const Images &images) const = 0;

  /**
   * @brief Trains the model using a list of images
   *
   * @param images The list of images
   * @param n_epoch The number of epochs
   * @param batch_size Minibatch size (set to 0 to disable)
   * @param learning_rate Learning rate of the model
   * @param verbose Whether to disable more or less information
   * @param checkpoint Save the model's parameter after a specific number of epochs (set
   * to 0 to disable)
   */
  virtual void fit(const Images &images,
                   int           n_epoch,
                   int           batch_size,
                   float         learning_rate,
                   bool          verbose,
                   int           checkpoint) = 0;
};

#endif
