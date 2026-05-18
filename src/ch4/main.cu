#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "check_prime_cpu.h"
#include "check_prime_gpu.cuh"
#include "cuda_utils.cuh"

int main() {
    constexpr long long start = 100'001LL;
    constexpr long long end = 190'001LL;

    cuda_utils::CudaEventTimer timer;
    timer.record_start();
    auto gpu_primes = ch4::check_prime_gpu_kernel(start, end);
    timer.record_stop();
    const float gpu_ms = timer.elapsed_ms();

    std::vector<long long> cpu_primes;
    const auto cpu_start = std::chrono::high_resolution_clock::now();
    for (long long num = start; num <= end; num += 2) {
        if (ch4::check_prime_cpu(num)) {
            cpu_primes.push_back(num);
        }
    }
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    std::sort(gpu_primes.begin(), gpu_primes.end());
    std::sort(cpu_primes.begin(), cpu_primes.end());
    const bool match = gpu_primes == cpu_primes;

    std::cout << "GPU time : " << gpu_ms << " ms\n"
              << "CPU time : " << cpu_ms.count() << " ms\n"
              << "Speedup  : " << cpu_ms.count() / gpu_ms << "x\n"
              << "Results  : " << (match ? "match" : "MISMATCH") << "\n";

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}
