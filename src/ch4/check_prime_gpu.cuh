#pragma once

namespace ch4 {

__device__ auto is_prime_device(long long n) -> bool;

// Launches a CUDA kernel that tests each integer in [start, end] for primality.
auto check_prime_gpu_kernel(long long start, long long end) -> void;

} // namespace ch4
