#include <gtest/gtest.h>

#include <vector>

#include "check_prime_cpu.h"
#include "check_prime_gpu.cuh"
#include "cuda_utils.cuh"

namespace {

__global__ void is_prime_batch_kernel(const long long* nums, char* results,
                                      int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = ch4::is_prime_device(nums[idx]) ? 1 : 0;
    }
}

void run_batch(const std::vector<long long>& nums, std::vector<bool>& results) {
    const int count = static_cast<int>(nums.size());
    cuda_utils::device_buffer<long long> d_nums(count);
    cuda_utils::device_buffer<char> d_results(count);
    cuda_utils::copy_to_device(d_nums, nums.data(), count);

    const dim3 block(256);
    const dim3 grid((count + 255) / 256);
    is_prime_batch_kernel<<<grid, block>>>(d_nums.data(), d_results.data(),
                                           count);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    std::vector<char> tmp(count);
    cuda_utils::copy_to_host(tmp.data(), d_results, count);
    for (int i = 0; i < count; ++i) {
        results[i] = (tmp[i] != 0);
    }
}

} // namespace

// ---- is_prime_device ----

TEST(IsPrimeDevice, MatchesCpuReferenceSmall) {
    std::vector<long long> nums;
    for (long long i = -2; i <= 100; ++i) {
        nums.push_back(i);
    }
    std::vector<bool> gpu_results(nums.size());
    run_batch(nums, gpu_results);

    for (std::size_t i = 0; i < nums.size(); ++i) {
        EXPECT_EQ(gpu_results[i], ch4::check_prime_cpu(nums[i]))
            << "num=" << nums[i];
    }
}

TEST(IsPrimeDevice, KnownPrimes) {
    const std::vector<long long> primes = {2, 3, 5, 7, 11, 13, 997, 999999937};
    std::vector<bool> results(primes.size());
    run_batch(primes, results);

    for (std::size_t i = 0; i < primes.size(); ++i) {
        EXPECT_TRUE(results[i]) << "expected prime: " << primes[i];
    }
}

TEST(IsPrimeDevice, KnownComposites) {
    const std::vector<long long> composites = {0, 1, 4, 9, 100, 561, 999999938};
    std::vector<bool> results(composites.size());
    run_batch(composites, results);

    for (std::size_t i = 0; i < composites.size(); ++i) {
        EXPECT_FALSE(results[i]) << "expected composite: " << composites[i];
    }
}
