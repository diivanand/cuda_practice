#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "cuda_utils.cuh"
#include "vec_add.cuh"

// ---- vec_add kernel ----

TEST(VecAdd, SmallKnownInput) {
    constexpr std::size_t n = 8;
    const std::vector<float> ha = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> hb = {10.0f, 20.0f, 30.0f, 40.0f,
                                   50.0f, 60.0f, 70.0f, 80.0f};
    std::vector<float> hout(n, 0.0f);

    cuda_utils::device_buffer<float> da(n), db(n), dout(n);
    cuda_utils::copy_to_device(da, ha.data(), n);
    cuda_utils::copy_to_device(db, hb.data(), n);

    ch1::vec_add<<<1, 32>>>(da.data(), db.data(), dout.data(),
                            static_cast<int>(n));
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    cuda_utils::copy_to_host(hout.data(), dout, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(hout[i], ha[i] + hb[i]) << "index " << i;
    }
}

TEST(VecAdd, AllZeros) {
    constexpr std::size_t n = 256;
    const std::vector<float> ha(n, 0.0f), hb(n, 0.0f);
    std::vector<float> hout(n, 1.0f);

    cuda_utils::device_buffer<float> da(n), db(n), dout(n);
    cuda_utils::copy_to_device(da, ha.data(), n);
    cuda_utils::copy_to_device(db, hb.data(), n);

    ch1::vec_add<<<1, 256>>>(da.data(), db.data(), dout.data(),
                             static_cast<int>(n));
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    cuda_utils::copy_to_host(hout.data(), dout, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(hout[i], 0.0f) << "index " << i;
    }
}

TEST(VecAdd, LargeIota) {
    constexpr std::size_t n = 1u << 20;
    std::vector<float> ha(n), hb(n, 2.0f), hout(n);
    std::iota(ha.begin(), ha.end(), 0.0f);

    cuda_utils::device_buffer<float> da(n), db(n), dout(n);
    cuda_utils::copy_to_device(da, ha.data(), n);
    cuda_utils::copy_to_device(db, hb.data(), n);

    const int ni = static_cast<int>(n);
    const dim3 block(256);
    const dim3 grid((ni + 255) / 256);
    ch1::vec_add<<<grid, block>>>(da.data(), db.data(), dout.data(), ni);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    cuda_utils::copy_to_host(hout.data(), dout, n);

    EXPECT_FLOAT_EQ(hout[0], 2.0f);
    EXPECT_FLOAT_EQ(hout[n / 2], static_cast<float>(n / 2) + 2.0f);
    EXPECT_FLOAT_EQ(hout[n - 1], static_cast<float>(n - 1) + 2.0f);
}
