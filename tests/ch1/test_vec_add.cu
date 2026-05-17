#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "cuda_utils.cuh"
#include "vec_add.cuh"

// ---- vec_add kernel ----

TEST(VecAdd, SmallKnownInput) {
    constexpr std::size_t n = 8;
    const std::vector<float> ha = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> hb = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    std::vector<float> hout(n, 0.0f);

    device_buffer<float> da(n), db(n), dout(n);
    copy_to_device(da, ha.data(), n);
    copy_to_device(db, hb.data(), n);

    vec_add<<<1, 32>>>(da.data(), db.data(), dout.data(), static_cast<int>(n));
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    copy_to_host(hout.data(), dout, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(hout[i], ha[i] + hb[i]) << "index " << i;
    }
}

TEST(VecAdd, AllZeros) {
    constexpr std::size_t n = 256;
    const std::vector<float> ha(n, 0.0f), hb(n, 0.0f);
    std::vector<float> hout(n, 1.0f);

    device_buffer<float> da(n), db(n), dout(n);
    copy_to_device(da, ha.data(), n);
    copy_to_device(db, hb.data(), n);

    vec_add<<<1, 256>>>(da.data(), db.data(), dout.data(), static_cast<int>(n));
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    copy_to_host(hout.data(), dout, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(hout[i], 0.0f) << "index " << i;
    }
}

TEST(VecAdd, LargeIota) {
    constexpr std::size_t n = 1u << 20;
    std::vector<float> ha(n), hb(n, 2.0f), hout(n);
    std::iota(ha.begin(), ha.end(), 0.0f);

    device_buffer<float> da(n), db(n), dout(n);
    copy_to_device(da, ha.data(), n);
    copy_to_device(db, hb.data(), n);

    const int ni = static_cast<int>(n);
    const dim3 block(256);
    const dim3 grid((ni + 255) / 256);
    vec_add<<<grid, block>>>(da.data(), db.data(), dout.data(), ni);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    copy_to_host(hout.data(), dout, n);

    EXPECT_FLOAT_EQ(hout[0], 2.0f);
    EXPECT_FLOAT_EQ(hout[n / 2], static_cast<float>(n / 2) + 2.0f);
    EXPECT_FLOAT_EQ(hout[n - 1], static_cast<float>(n - 1) + 2.0f);
}

// ---- device_buffer ----

TEST(DeviceBuffer, DefaultConstruct) {
    const device_buffer<float> buf;
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.bytes(), 0u);
    EXPECT_FALSE(static_cast<bool>(buf));
}

TEST(DeviceBuffer, AllocateSizeAndBytes) {
    constexpr std::size_t n = 64;
    const device_buffer<float> buf(n);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), n);
    EXPECT_EQ(buf.bytes(), n * sizeof(float));
    EXPECT_TRUE(static_cast<bool>(buf));
}

TEST(DeviceBuffer, MoveConstruct) {
    device_buffer<float> src(32);
    const float* original_ptr = src.data();

    const device_buffer<float> dst(std::move(src));

    EXPECT_EQ(dst.data(), original_ptr);
    EXPECT_EQ(dst.size(), 32u);
}

TEST(DeviceBuffer, MoveAssignReplaces) {
    device_buffer<float> src(32);
    const float* original_ptr = src.data();

    device_buffer<float> dst(16);
    dst = std::move(src);

    EXPECT_EQ(dst.data(), original_ptr);
    EXPECT_EQ(dst.size(), 32u);
}

TEST(DeviceBuffer, Reset) {
    device_buffer<float> buf(64);
    buf.reset();
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_FALSE(static_cast<bool>(buf));
}
