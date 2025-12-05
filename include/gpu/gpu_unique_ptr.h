#ifndef GPU_UNIQUE_PTR_H
#define GPU_UNIQUE_PTR_H

/**
 * @brief Class that represents a pointer on the GPU
 *
 */
class Gpu_Unique_Ptr {
  /**
   * @brief The actual pointer
   *
   */
  float *_ptr;

public:
  /**
   * @brief Default constructor, does not allocate anything
   *
   */
  Gpu_Unique_Ptr();
  /**
   * @brief Allocate elements on GPU
   *
   * @param n Number of elements
   */
  Gpu_Unique_Ptr(unsigned int n);
  /**
   * @brief Take ownership from another unique_ptr
   *
   * @param other The other pointer
   */
  Gpu_Unique_Ptr(Gpu_Unique_Ptr &&other);
  /**
   * @brief Take ownership from another unique_ptr
   *
   * @param other The other pointer
   * @return Gpu_Unique_Ptr& Reference to current pointer
   */
  Gpu_Unique_Ptr &operator=(Gpu_Unique_Ptr &&other);
  /**
   * @brief Destroy the Gpu_Unique_Ptr object
   *
   */
  ~Gpu_Unique_Ptr();

  // Remove copy constructors
  Gpu_Unique_Ptr(const Gpu_Unique_Ptr &)            = delete;
  Gpu_Unique_Ptr &operator=(const Gpu_Unique_Ptr &) = delete;

  /**
   * @brief Get the actual pointer
   *
   * @return float* The pointer
   */
  float *get() const noexcept;
  /**
   * @brief Deallocate current pointer
   *
   */
  void reset();
};

/**
 * @brief Class that represents a pinned memory
 *
 */
class Pinned_Unique_Ptr {
  float *_ptr;

public:
  /**
   * @brief Default constructor, does not allocate anything
   *
   */
  Pinned_Unique_Ptr();
  /**
   * @brief Allocate pinned memory
   *
   * @param n Number of elements
   */
  Pinned_Unique_Ptr(unsigned int n);
  /**
   * @brief Take ownership from another unique_ptr
   *
   * @param other The other pointer
   */
  Pinned_Unique_Ptr(Pinned_Unique_Ptr &&other);
  /**
   * @brief Take ownership from another unique_ptr
   *
   * @param other The other pointer
   * @return Gpu_Unique_Ptr& Reference to current pointer
   */
  Pinned_Unique_Ptr &operator=(Pinned_Unique_Ptr &&other);
  /**
   * @brief Destroy the Gpu_Unique_Ptr object
   *
   */
  ~Pinned_Unique_Ptr();

  // Remove copy constructors
  Pinned_Unique_Ptr(const Pinned_Unique_Ptr &)            = delete;
  Pinned_Unique_Ptr &operator=(const Pinned_Unique_Ptr &) = delete;

  /**
   * @brief Get the actual pointer
   *
   * @return float* The pointer
   */
  float *get() const noexcept;
  /**
   * @brief Deallocate current pointer
   *
   */
  void reset();
};

#endif