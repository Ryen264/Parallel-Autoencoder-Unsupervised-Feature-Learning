#ifndef CPU_AUTOENCODER_H
#define CPU_AUTOENCODER_H

#include "autoencoder.h"

/**
 * @brief The class that encapsulates the network using CPU
 *
 */
class Cpu_Autoencoder : IAutoencoder {
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
  ~Cpu_Autoencoder() override = default;

  /**
   * @brief Encodes a list of images
   *
   * @param images The images to be encoded
   * @return Images The encoded images
   */
  Images encode(const Images &images) const override;
  /**
   * @brief Decodes a list of images
   *
   * @param images The images to be decoded
   * @return Images The decoded images
   */
  Images decode(const Images &images) const override;

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
  void fit(const Images &images,
           int           n_epoch,
           int           batch_size,
           float         learning_rate,
           bool          verbose,
           int           checkpoint) override;
};

#endif