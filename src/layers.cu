#include "cpu_layers.h"
#include "layers.h"

void conv2D(float *in,
            float *filter,
            float *out,
            int    n,
            int    depth,
            int    n_filter,
            bool   use_device) {
  if (use_device) {
    // TODO
  } else
    cpu_conv2D(in, filter, out, n, depth, n_filter);
}

void relu(float *in, float *out, int n, int depth, bool use_device) {
  if (use_device) {
    // TODO
  } else
    cpu_relu(in, out, n, depth);
}

void max_pooling(float *in, float *out, int n, int depth, bool use_device) {
  if (use_device) {
    // TODO
  } else
    cpu_max_pooling(in, out, n, depth);
}

void upsampling(float *in, float *out, int n, int depth, bool use_device) {
  if (use_device) {
    // TODO
  } else
    cpu_upsampling(in, out, n, depth);
}

float mse_loss(float *expected, float *actual, int n, int depth, bool use_device) {
  if (use_device) {
    // TODO
    return 0.0;
  }

  return cpu_mse_loss(expected, actual, n, depth);
}