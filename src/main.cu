#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

#include "cuda_utils.cuh"
#include "vec_add.cuh"

static void launch_vec_add(const float* a, const float* b, float* out, std::size_t n) {
    const int ni = static_cast<int>(n);

    dim3 block(256);
    dim3 grid((ni + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));

    vec_add<<<grid, block>>>(a, b, out, ni);
    CUDA_TRY(cudaGetLastError());
}

int main() {
    constexpr std::size_t n = 1u << 20;

    std::vector<float> ha(n), hb(n), hout(n);
    std::iota(ha.begin(), ha.end(), 0.0f);
    std::fill(hb.begin(), hb.end(), 2.0f);

    device_buffer<float> da(n), db(n), dout(n);

    copy_to_device(da, ha.data(), n);
    copy_to_device(db, hb.data(), n);

    launch_vec_add(da.data(), db.data(), dout.data(), n);
    CUDA_TRY(cudaDeviceSynchronize());

    copy_to_host(hout.data(), dout, n);

    std::cout << "out[0]   = " << hout[0] << "\n";
    std::cout << "out[n-1] = " << hout[n - 1] << "\n";
    return 0;
}
