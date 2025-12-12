#ifndef GPU_LAYERS_H
#define GPU_LAYERS_H

/**
 * @brief 2D convolution layer on GPU
 */
__global__ void gpu_conv2D(float *in,
                           float *filter,
                           float *out,
                           int    width,
                           int    height,
                           int    depth,
                           int    n_filter);

/**
 * @brief Apply bias on GPU
 */
__global__ void
gpu_add_bias(float *in, float *bias, float *out, int width, int height, int depth);

/**
 * @brief ReLU activation on GPU
 */
__global__ void gpu_relu(float *in, float *out, int width, int height, int depth);

/**
 * @brief Max pooling (downsample 2x) on GPU
 */
__global__ void
gpu_max_pooling(float *in, float *out, int width, int height, int depth);

/**
 * @brief Upsampling (upsample 2x) on GPU
 */
__global__ void gpu_upsampling(float *in, float *out, int width, int height, int depth);

/**
 * @brief Mean squared error loss on GPU
 */
__global__ void gpu_mse_loss(
    float *expected, float *actual, float *out, int width, int height, int depth);

/**
 * @brief MSE gradient on GPU
 */
__global__ void gpu_mse_grad(
    float *expected, float *actual, float *d_out, int width, int height, int depth);

/**
 * @brief ReLU backward on GPU
 */
__global__ void gpu_relu_backward(
    float *in, float *d_out, float *d_in, int width, int height, int depth);

/**
 * @brief Max pooling backward on GPU
 */
__global__ void gpu_max_pooling_backward(
    float *in, float *d_out, float *d_in, int width, int height, int depth);

/**
 * @brief Upsampling backward on GPU
 */
__global__ void
gpu_upsampling_backward(float *d_out, float *d_in, int width, int height, int depth);

/**
 * @brief Bias gradient calculation on GPU
 */
__global__ void
gpu_bias_grad(float *d_out, float *d_bias, int width, int height, int depth);

/**
 * @brief Conv2D gradient calculation on GPU
 */
__global__ void gpu_conv2D_grad(float *in,
                                float *d_out,
                                float *d_filter,
                                int    width,
                                int    height,
                                int    depth,
                                int    n_filter);

/**
 * @brief Update weights using gradient descent on GPU
 */
__global__ void
gpu_update_weight(float *weight, float *gradient, int size, float learning_rate);

#endif // GPU_LAYERS_H
