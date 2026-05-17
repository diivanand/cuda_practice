#pragma once
#include <cuda_runtime.h>

// Element-wise float addition: out[i] = a[i] + b[i] for i in [0, n).
// Threads with i >= n are no-ops; launch with enough blocks to cover n elements.
__global__ void vec_add(const float* a, const float* b, float* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = a[i] + b[i];
}
