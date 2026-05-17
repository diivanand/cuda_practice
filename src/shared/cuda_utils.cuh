#pragma once
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <type_traits>

// ---- Error handling ----

namespace cuda_utils {

// Prints a CUDA error message and terminates the process. Called by CUDA_TRY.
[[noreturn]] inline auto cuda_fail(cudaError_t e,
                                   std::string_view what,
                                   std::string_view file,
                                   int line) -> void {
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
    if (_e != cudaSuccess) {                                                   \
      cuda_utils::cuda_fail(_e, #expr, __FILE__, __LINE__);                   \
    }                                                                          \
  } while (0)

// ---- RAII device buffer ----

namespace cuda_utils {

// RAII wrapper for a cudaMalloc'd device allocation. Non-copyable; move-only.
// Zero-count construction is valid and allocates nothing.
template <class T>
class device_buffer {
  static_assert(!std::is_void_v<T>, "device_buffer<void> is not allowed");

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
  auto operator=(const device_buffer&) -> device_buffer& = delete;

  device_buffer(device_buffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  auto operator=(device_buffer&& other) noexcept -> device_buffer& {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  auto data() noexcept -> T* { return ptr_; }
  auto data() const noexcept -> const T* { return ptr_; }
  auto size() const noexcept -> std::size_t { return count_; }
  auto bytes() const noexcept -> std::size_t { return count_ * sizeof(T); }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  // Frees the allocation and zeroes the pointer and count. Safe to call multiple times.
  // noexcept: reset() is called from the destructor, where throwing is not an
  // option. cudaFree errors are intentionally ignored — best-effort cleanup,
  // consistent with standard RAII practice.
  auto reset() noexcept -> void {
    if (ptr_) {
      cudaFree(ptr_);
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
inline auto copy_to_device(device_buffer<T>& dst, const T* src, std::size_t count) -> void {
  assert(count <= dst.size());
  if (count == 0) { return; }
  CUDA_TRY(cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Copies count elements from src into host pointer dst. count == 0 is a no-op.
template <class T>
inline auto copy_to_host(T* dst, const device_buffer<T>& src, std::size_t count) -> void {
  assert(count <= src.size());
  if (count == 0) { return; }
  CUDA_TRY(cudaMemcpy(dst, src.data(), count * sizeof(T), cudaMemcpyDeviceToHost));
}

// ---- CUDA event timer ----

class CudaEventTimer {
public:
    CudaEventTimer() {
        CUDA_TRY(cudaEventCreate(&start_));
        CUDA_TRY(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() {
        if (start_) { cudaEventDestroy(start_); }
        if (stop_)  { cudaEventDestroy(stop_); }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    auto operator=(const CudaEventTimer&) -> CudaEventTimer& = delete;

    CudaEventTimer(CudaEventTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_  = nullptr;
    }

    auto operator=(CudaEventTimer&& other) noexcept -> CudaEventTimer& {
        if (this != &other) {
            if (start_) { cudaEventDestroy(start_); }
            if (stop_)  { cudaEventDestroy(stop_); }
            start_ = other.start_;
            stop_  = other.stop_;
            other.start_ = nullptr;
            other.stop_  = nullptr;
        }
        return *this;
    }

    auto record_start() -> void { CUDA_TRY(cudaEventRecord(start_, 0)); }
    auto record_stop() -> void {
        CUDA_TRY(cudaEventRecord(stop_, 0));
        CUDA_TRY(cudaEventSynchronize(stop_));
    }

    auto elapsed_ms() const -> float {
        float ms = 0.0F;
        CUDA_TRY(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

} // namespace cuda_utils
