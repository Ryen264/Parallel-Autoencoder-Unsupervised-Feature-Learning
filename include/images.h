#ifndef IMAGE_H
#define IMAGE_H

// Use unique_ptr to prevent memory leak
#include <memory>

using std::move;
using std::unique_ptr, std::make_unique;

/**
 * @brief Struct that represents a list of image (either encoded or decoded)
 *
 */
struct Images {
  // The list of images (flattened)
  unique_ptr<float[]> img;
  // The number of images in the list
  int n;
  // The width of the image
  int width;
  // The bit-depth of the image
  int depth;

  /**
   * @brief Create a list of unitialized image
   *
   * @param n The number of images
   * @param width The width of the images
   * @param depth The depth of the images
   */
  Images(int n, int width, int depth)
      : img(make_unique<float[]>(n * width * width * depth))
      , n(n)
      , width(width)
      , depth(depth) {};

  /**
   * @brief Initializes a list of images
   *
   * @param img The flattened images
   * @param n The number of images
   * @param width The width of the images
   * @param depth The depth of the images
   */
  Images(unique_ptr<float[]> &img, int n, int width, int depth)
      : img(move(img)), n(n), width(width), depth(depth) {};

  /**
   * @brief Get a raw pointer to the first image
   *
   * @return float* The pointer
   */
  float *get() const noexcept { return img.get(); }
};

#endif