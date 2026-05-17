#include "check_prime_gpu.cuh"

#include <cstdio>
#include <cuda_runtime.h>

namespace ch4 {

__device__ auto is_prime_device(const long long n) -> bool {
    if (n < 2) {
        return false;
    }
    if (n == 2) {
        return true;
    }
    if (n % 2 == 0) {
        return false;
    }
    for (long long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

__global__ auto prime_range_kernel(const long long start, const long long end)
    -> void {
    const long long num =
        start + static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (num <= end && is_prime_device(num)) {
        printf("%lld\n", num);
    }
}

auto check_prime_gpu_kernel(long long start, long long end) -> void {
    const long long count = end - start + 1;
    constexpr int block = 256;
    const auto grid = static_cast<unsigned>((count + block - 1) / block);
    prime_range_kernel<<<grid, block>>>(start, end);
    cudaDeviceSynchronize();
}

} // namespace ch4
