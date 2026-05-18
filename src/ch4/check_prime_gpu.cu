#include "check_prime_gpu.cuh"

#include <cuda_runtime.h>
#include <vector>

#include "cuda_utils.cuh"

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

// Each thread atomically appends its prime directly into out_primes, avoiding
// a separate bool pass and a host-side compaction loop.
__global__ auto prime_range_kernel(const long long start, const long long end,
                                   long long* out_primes, int* out_count) -> void {
    const long long tid =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long num = start + tid * 2;
    if (num <= end && is_prime_device(num)) {
        const int idx = atomicAdd(out_count, 1);
        out_primes[idx] = num;
    }
}

auto check_prime_gpu_kernel(long long start, long long end)
    -> std::vector<long long> {
    const auto num_candidates = static_cast<std::size_t>((end - start) / 2 + 1);
    constexpr int block = 256;
    const auto grid = static_cast<unsigned>((num_candidates + block - 1) / block);

    cuda_utils::device_buffer<long long> d_primes(num_candidates);
    cuda_utils::device_buffer<int> d_count(1);
    CUDA_TRY(cudaMemset(d_count.data(), 0, sizeof(int)));

    prime_range_kernel<<<grid, block>>>(start, end, d_primes.data(), d_count.data());
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    // Two small transfers: count first, then only the compacted prime values.
    int prime_count = 0;
    cuda_utils::copy_to_host(&prime_count, d_count, 1);

    std::vector<long long> primes(static_cast<std::size_t>(prime_count));
    cuda_utils::copy_to_host(primes.data(), d_primes,
                             static_cast<std::size_t>(prime_count));
    return primes;
}

} // namespace ch4
