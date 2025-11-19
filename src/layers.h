#ifndef LAYERS_H
#define LAYERS_H

/**
 * @brief 2D convolution layer with padding=1 and stride=1
 *
 * @param in The input array
 * @param filter The layer parameters
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 * @param n_filter The number of filters
 * @param use_device Choose to perform the calculations on CPU (false) or GPU (true)
 */
void conv2D(float *in,
            float *filter,
            float *out,
            int    n,
            int    depth,
            int    n_filter,
            bool   use_device);

/**
 * @brief ReLU layer
 *
 * @param in The input array
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 * @param use_device Choose to perform the calculations on CPU (false) or GPU (true)
 */
void relu(float *in, float *out, int n, int depth, bool use_device);

/**
 * @brief Max pooling layer to downsample by half
 *
 * @param in The input array
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 * @param use_device Choose to perform the calculations on CPU (false) or GPU (true)
 */
void max_pooling(float *in, float *out, int n, int depth, bool use_device);

/**
 * @brief Use nearest neighbor interpolation to double spatial dimension
 *
 * @param in The input array
 * @param out The output array
 * @param n The width of the input array
 * @param depth The depth of the input array
 * @param use_device Choose to perform the calculations on CPU (false) or GPU (true)
 */
void upsampling(float *in, float *out, int n, int depth, bool use_device);

/**
 * @brief Calculate the mean squared error between the output and the target
 *
 * @param expected The target array
 * @param actual The array to calculate MSE
 * @param n The width of the array
 * @param depth The depth of the array
 * @param use_device Choose to perform the calculations on CPU (false) or GPU (true)
 *
 * @return The MSE value
 */
float mse_loss(float *expected, float *actual, int n, int depth, bool use_device);

#endif
