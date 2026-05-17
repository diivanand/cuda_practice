//
// Created by diiv on 5/16/26.
//

#ifndef CUDA_PRACTICE_CHECK_PRIME_GPU_CUH
#define CUDA_PRACTICE_CHECK_PRIME_GPU_CUH
// Launches a CUDA kernel that tests each integer in [start, end] for primality.
void check_prime_gpu_kernel(long long start, long long end);
#endif //CUDA_PRACTICE_CHECK_PRIME_GPU_CUH
