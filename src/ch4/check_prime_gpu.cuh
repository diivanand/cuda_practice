#pragma once

namespace ch4 {

// Launches a CUDA kernel that tests each integer in [start, end] for primality.
auto check_prime_gpu_kernel(long long start, long long end) -> void;

} // namespace ch4
