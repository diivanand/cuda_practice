#pragma once

#include <vector>

namespace ch4 {

__device__ auto is_prime_device(long long n) -> bool;

// Finds all primes in [start, end] (odd candidates only). Returns them sorted.
auto check_prime_gpu(long long start, long long end) -> std::vector<long long>;

} // namespace ch4
