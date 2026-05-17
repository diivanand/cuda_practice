#include "vec_add.cuh"

namespace ch1 {

__global__ auto vec_add(const float* a, const float* b, float* out, int n) -> void {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { out[i] = a[i] + b[i]; }
}

} // namespace ch1
