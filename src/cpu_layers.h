#ifndef CPU_LAYERS_H
#define CPU_LAYERS_H

constexpr int CONV_FILTER_SIZE = 3;

/**
 * @brief 2D convolution layer with padding=1 and stride=1
 *
 * @param in The input array
 * @param filter The layer parameters
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 * @param n_filter The number of filters
 */
void cpu_conv2D(float *in, float *filter, float *out, int n, int depth, int n_filter);

/**
 * @brief ReLU layer
 *
 * @param in The input array
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 */
void cpu_relu(float *in, float *out, int n, int depth);

/**
 * @brief Max pooling layer to downsample by half
 *
 * @param in The input array
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 */
void cpu_max_pooling(float *in, float *out, int n, int depth);

/**
 * @brief Use nearest neighbor interpolation to double spatial dimension
 *
 * @param in The input array
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 */
void cpu_upsampling(float *in, float *out, int n, int depth);

/**
 * @brief Calculate the mean squared error between the output and the target
 *
 * @param expected The target array
 * @param actual The array to calculate MSE
 * @param n The width of the array
 * @param depth The depth of the array
 *
 * @return The MSE value
 */
float cpu_mse_loss(float *expected, float *actual, int n, int depth);

#endif
