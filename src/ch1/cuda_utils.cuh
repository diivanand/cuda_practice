#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <type_traits>

// ---- Error handling ----

namespace cuda_utils {

// Prints a CUDA error message and terminates the process. Called by CUDA_TRY.
[[noreturn]] inline void cuda_fail(cudaError_t e,
                                   std::string_view what,
                                   std::string_view file,
                                   int line) {
  std::cerr << "CUDA error: " << what
            << " | " << cudaGetErrorString(e)
            << " (" << static_cast<int>(e) << ")"
            << " @ " << file << ":" << line << "\n";
  std::exit(1);
}

} // namespace cuda_utils

// Evaluates expr, which must return cudaError_t. Calls cuda_fail (process exit) on failure.
#define CUDA_TRY(expr)                                                         \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess)                                                     \
      cuda_utils::cuda_fail(_e, #expr, __FILE__, __LINE__);                   \
  } while (0)

// ---- RAII device buffer ----

namespace cuda_utils {

// RAII wrapper for a cudaMalloc'd device allocation. Non-copyable; move-only.
// Zero-count construction is valid and allocates nothing.
template <class T>
class device_buffer {
  static_assert(!std::is_void<T>::value, "device_buffer<void> is not allowed");

public:
  using value_type = T;

  device_buffer() = default;

  explicit device_buffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_TRY(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  ~device_buffer() { reset(); }

  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  device_buffer(device_buffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  device_buffer& operator=(device_buffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return count_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  // Frees the allocation and zeroes the pointer and count. Safe to call multiple times.
  void reset() noexcept {
    if (ptr_) {
      cudaFree(ptr_); // best-effort in destructor context
      ptr_ = nullptr;
      count_ = 0;
    }
  }

private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// ---- memcpy helpers ----

// Copies count elements from host pointer src into dst. count == 0 is a no-op.
template <class T>
inline void copy_to_device(device_buffer<T>& dst, const T* src, std::size_t count) {
  if (count == 0) return;
  CUDA_TRY(cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Copies count elements from src into host pointer dst. count == 0 is a no-op.
template <class T>
inline void copy_to_host(T* dst, const device_buffer<T>& src, std::size_t count) {
  if (count == 0) return;
  CUDA_TRY(cudaMemcpy(dst, src.data(), count * sizeof(T), cudaMemcpyDeviceToHost));
}

} // namespace cuda_utils
