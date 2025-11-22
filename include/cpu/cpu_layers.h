#ifndef CPU_LAYERS_H
#define CPU_LAYERS_H

/**
 * @brief 2D convolution layer with padding=1 and stride=1
 *
 * @param in The input array
 * @param filter The layer parameters
 * @param out The output array
 * @param n The number of images to apply filter
 * @param width The width of the input array
 * @param depth The depth of the input array
 * @param n_filter The number of filters
 */
void cpu_conv2D(
    float *in, float *filter, float *out, int n, int width, int depth, int n_filter);

/**
 * @brief Apply bias into images
 *
 * @param in The input images
 * @param bias The bias to be added
 * @param out The output images
 * @param n The number of images
 * @param width The width of the images
 * @param depth The depth of the images
 */
void cpu_add_bias(float *in, float *bias, float *out, int n, int width, int depth);

/**
 * @brief ReLU layer
 *
 * @param in The input array
 * @param out The output array
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param depth The depth of the input array
 */
void cpu_relu(float *in, float *out, int n, int width, int depth);

/**
 * @brief Max pooling layer to downsample by half
 *
 * @param in The input array
 * @param out The output array
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param depth The depth of the input array
 */
void cpu_max_pooling(float *in, float *out, int n, int width, int depth);

/**
 * @brief Use nearest neighbor interpolation to double spatial dimension
 *
 * @param in The input array
 * @param out The output array
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param depth The depth of the input array
 */
void cpu_upsampling(float *in, float *out, int n, int width, int depth);

/**
 * @brief Calculate the mean squared error between the output and the target
 *
 * @param expected The target array
 * @param actual The array to calculate MSE
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param depth The depth of the array
 *
 * @return float The MSE value
 */
float cpu_mse_loss(float *expected, float *actual, int n, int width, int depth);

/**
 * @brief Calculate gradiant of the loss function
 *
 * @param expected The target array
 * @param actual The array to calculate MSE
 * @param d_out Array to pass delta (backward propagation)
 * @param n The number of images in the array
 * @param width The width of the input array
 * @param depth The depth of the array
 */
void
cpu_mse_grad(float *expected, float *actual, float *d_out, int n, int width, int depth);

/**
 * @brief Backward propagation for ReLU
 *
 * @param in The input data
 * @param d_out The output delta
 * @param d_in The input delta
 * @param n The number of images
 * @param width The width of the images
 * @param depth The depth of the images
 */
void
cpu_relu_backward(float *in, float *d_out, float *d_in, int n, int width, int depth);

/**
 * @brief Backward propagation for max pooling
 *
 * @param in The input data
 * @param d_out The output delta
 * @param d_in The input delta
 * @param n The number of images
 * @param width The width of the images
 * @param depth The depth of the images
 */
void cpu_max_pooling_backward(
    float *in, float *d_out, float *d_in, int n, int width, int depth);

/**
 * @brief Backward propagation for upsampling
 *
 * @param d_out The output delta
 * @param d_in The input delta
 * @param n The number of images
 * @param width The width of the images
 * @param depth The depth of the images
 */
void cpu_upsampling_backward(float *d_out, float *d_in, int n, int width, int depth);

/**
 * @brief Calculate gradient for conv2D bias
 *
 * @param d_out The output delta
 * @param grad_bias The gradient output
 * @param n Number of images
 * @param width Width of the images
 * @param depth Depth of the images
 */
void cpu_bias_grad(float *d_out, float *grad_bias, int n, int width, int depth);

/**
 * @brief Calculate gradient for conv2D filter
 *
 * @param in The input images
 * @param d_out The output delta
 * @param grad_filter The gradient output
 * @param n Number of images
 * @param width Width of the images
 * @param depth Depth of the images
 * @param n_filter Number of filters
 */
void cpu_conv2D_grad(float *in,
                     float *d_out,
                     float *grad_filter,
                     int    n,
                     int    width,
                     int    depth,
                     int    n_filter);

/**
 * @brief Update the weights of the array
 *
 * @param in The input array
 * @param grad The gradient
 * @param size The size of the array
 * @param learning_rate The learning
 */
void cpu_update_weight(float *in, float *grad, int size, float learning_rate);

#endif
