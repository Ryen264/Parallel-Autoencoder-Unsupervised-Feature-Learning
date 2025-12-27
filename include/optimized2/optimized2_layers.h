#ifndef OPTIMIZED2_LAYERS_H
#define OPTIMIZED2_LAYERS_H

#include "constants.h"
#include "macro.h"

void optimized2_full_filter(float *in,
                            float *filter,
                            float *bias,
                            float *in_relu,
                            float *out,
                            int    n,
                            int    width,
                            int    height,
                            int    depth,
                            int    n_filter,
                            dim3   block_size);

/**
 * @brief Max pooling layer to downsample by half
 *
 * @param in The input array
 * @param out The output array
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param height The height of the input array
 * @param depth The depth of the input array
 * @param block_size The block size to call the kernel functions
 */
void optimized2_max_pooling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size);

/**
 * @brief Upsampling layer to upsample by twice
 *
 * @param in The input array
 * @param out The output array
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param height The height of the input array
 * @param depth The depth of the input array
 * @param block_size The block size to call the kernel functions
 */
void optimized2_upsampling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size);

/**
 * @brief Mean squared error loss
 *
 * @param expected The expected result
 * @param actual The actual result
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 * @return float The MSE loss
 */
float optimized2_mse_loss(float *expected,
                          float *actual,
                          int    n,
                          int    width,
                          int    height,
                          int    depth,
                          dim3   block_size);

/**
 * @brief Mean squared error gradient
 *
 * @param expected The expected result
 * @param actual The actual result
 * @param d_out The output array (delta_out)
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 */
void optimized2_mse_grad(float *expected,
                         float *actual,
                         float *d_out,
                         int    n,
                         int    width,
                         int    height,
                         int    depth,
                         dim3   block_size);

void optimized2_full_filter_grad(float *in,
                                 float *d_out,
                                 float *d_bias,
                                 float *d_filter,
                                 int    n,
                                 int    width,
                                 int    height,
                                 int    depth,
                                 int    n_filter,
                                 dim3   block_size);

/**
 * @brief ReLU layer backwards pass
 *
 * @param in The input from the forward pass
 * @param d_out The output gradient
 * @param d_in The incoming gradient
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 */
void optimized2_relu_backward(float *in,
                              float *d_out,
                              float *d_in,
                              int    n,
                              int    width,
                              int    height,
                              int    depth,
                              dim3   block_size);

/**
 * @brief Max pooling backwards pass
 *
 * @param in The input from the forward pass
 * @param d_out The output gradient
 * @param d_in The incoming gradient
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 */
void optimized2_max_pooling_backward(float *in,
                                     float *d_out,
                                     float *d_in,
                                     int    n,
                                     int    width,
                                     int    height,
                                     int    depth,
                                     dim3   block_size);

/**
 * @brief Upsampling backwards pass
 *
 * @param d_out The output gradient
 * @param d_in The incoming gradient
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 */
void optimized2_upsampling_backward(float *d_out,
                                    float *d_in,
                                    int    n,
                                    int    width,
                                    int    height,
                                    int    depth,
                                    dim3   block_size);

/**
 * @brief Update weights using gradient descent
 *
 * @param weight The weights to update
 * @param gradient The gradient
 * @param size The size of the arrays
 * @param learning_rate The learning rate
 * @param block_size The block size to call the kernel functions
 */
void optimized2_update_weight(
    float *weight, float *gradient, int size, float learning_rate, dim3 block_size);

void optimized2_conv2D_backward(float *d_out,
                                float *filter,
                                float *d_in,
                                int    n,
                                int    width,
                                int    height,
                                int    depth,
                                int    n_filter,
                                dim3   block_size);

#endif