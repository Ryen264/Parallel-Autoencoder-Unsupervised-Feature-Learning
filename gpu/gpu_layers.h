#ifndef GPU_LAYERS_H
#define GPU_LAYERS_H

#include <cuda_runtime.h>

/**
 * @brief 2D convolution layer on GPU
 */
void gpu_conv2D(float *in, float *filter, float *out, 
                int n, int width, int height, int depth, int n_filter);

/**
 * @brief Apply bias on GPU
 */
void gpu_add_bias(float *in, float *bias, float *out,
                  int n, int width, int height, int depth);

/**
 * @brief ReLU activation on GPU
 */
void gpu_relu(float *in, float *out,
              int n, int width, int height, int depth);

/**
 * @brief Max pooling (downsample 2x) on GPU
 */
void gpu_max_pooling(float *in, float *out,
                     int n, int width, int height, int depth);

/**
 * @brief Upsampling (upsample 2x) on GPU
 */
void gpu_upsampling(float *in, float *out,
                    int n, int width, int height, int depth);

/**
 * @brief Mean squared error loss on GPU
 */
float gpu_mse_loss(float *expected, float *actual,
                   int n, int width, int height, int depth);

/**
 * @brief MSE gradient on GPU
 */
void gpu_mse_grad(float *expected, float *actual, float *d_out,
                  int n, int width, int height, int depth);

/**
 * @brief ReLU backward on GPU
 */
void gpu_relu_backward(float *in, float *d_in, float *d_out,
                       int n, int width, int height, int depth);

/**
 * @brief Max pooling backward on GPU
 */
void gpu_max_pooling_backward(float *in, float *d_in, float *d_out,
                              int n, int width, int height, int depth);

/**
 * @brief Upsampling backward on GPU
 */
void gpu_upsampling_backward(float *d_in, float *d_out,
                             int n, int width, int height, int depth);

/**
 * @brief Bias gradient calculation on GPU
 */
void gpu_bias_grad(float *d_in, float *d_bias,
                   int n, int width, int height, int depth);

/**
 * @brief Conv2D gradient calculation on GPU
 */
void gpu_conv2D_grad(float *in, float *d_out, float *d_filter,
                     int n, int width, int height, int depth, int n_filter);

/**
 * @brief Update weights using gradient descent on GPU
 */
void gpu_update_weight(float *weight, float *gradient,
                       int size, float learning_rate);

#endif // GPU_LAYERS_H
