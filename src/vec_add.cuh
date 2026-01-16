#pragma once
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = a[i] + b[i];
}
