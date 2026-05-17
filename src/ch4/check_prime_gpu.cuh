#pragma once

// Launches a CUDA kernel that tests each integer in [start, end] for primality.
void check_prime_gpu_kernel(long long start, long long end);
