#ifndef OPTIMIZED1_LAYERS_H
#define OPTIMIZED1_LAYERS_H

#include "constants.h"
#include "macro.h"

/**
 * @brief 2D convolution layer with padding=1 and stride=1
 *
 * @param in The input array
 * @param filter The layer parameters
 * @param out The output array
 * @param n The number of images to apply filter
 * @param width The width of the input array
 * @param height The height of the input array
 * @param depth The depth of the input array
 * @param n_filter The number of filters
 * @param block_size The block size to call the kernel functions
 */
void optimized1_conv2D(float *in,
                       float *filter,
                       float *out,
                       int    n,
                       int    width,
                       int    height,
                       int    depth,
                       int    n_filter,
                       dim3   block_size);

/**
 * @brief Apply bias into images
 *
 * @param in The input images
 * @param bias The bias to be added
 * @param out The output images
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 */
void optimized1_add_bias(float *in,
                         float *bias,
                         float *out,
                         int    n,
                         int    width,
                         int    height,
                         int    depth,
                         dim3   block_size);

/**
 * @brief ReLU layer
 *
 * @param in The input array
 * @param out The output array
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param height The height of the input array
 * @param depth The depth of the input array
 * @param block_size The block size to call the kernel functions
 */
void optimized1_relu(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size);

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
void optimized1_avg_pooling(
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
void optimized1_upsampling(
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
float optimized1_mse_loss(float *expected,
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
void optimized1_mse_grad(float *expected,
                         float *actual,
                         float *d_out,
                         int    n,
                         int    width,
                         int    height,
                         int    depth,
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
void optimized1_relu_backward(float *in,
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
void optimized1_avg_pooling_backward(float *d_out,
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
void optimized1_upsampling_backward(float *d_out,
                                    float *d_in,
                                    int    n,
                                    int    width,
                                    int    height,
                                    int    depth,
                                    dim3   block_size);

/**
 * @brief Bias gradient calculation
 *
 * @param d_out The output delta
 * @param d_bias The bias gradient output
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param block_size The block size to call the kernel functions
 */
void optimized1_bias_grad(float *d_out,
                          float *d_bias,
                          int    n,
                          int    width,
                          int    height,
                          int    depth,
                          dim3   block_size);

/**
 * @brief Conv2D gradient calculation
 *
 * @param in The input from forward pass
 * @param d_out The incoming gradient
 * @param d_filter The filter gradient output
 * @param n The number of images
 * @param width The width of the images
 * @param height The height of the images
 * @param depth The depth of the images
 * @param n_filter The number of filters
 * @param block_size The block size to call the kernel functions
 */
void optimized1_conv2D_grad(float *in,
                            float *d_out,
                            float *d_filter,
                            int    n,
                            int    width,
                            int    height,
                            int    depth,
                            int    n_filter,
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
void optimized1_update_weight(
    float *weight, float *gradient, int size, float learning_rate, dim3 block_size);

void optimized1_conv2D_backward(float *d_out,
                                float *filter,
                                float *d_in,
                                int    n,
                                int    width,
                                int    height,
                                int    depth,
                                int    n_filter,
                                dim3   block_size);

#endif