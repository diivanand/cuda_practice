#include <gtest/gtest.h>

#include <cstddef>

#include "cuda_utils.cuh"

// ---- device_buffer ----

TEST(DeviceBuffer, DefaultConstruct) {
    const cuda_utils::device_buffer<float> buf;
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.bytes(), 0u);
    EXPECT_FALSE(static_cast<bool>(buf));
}

TEST(DeviceBuffer, AllocateSizeAndBytes) {
    constexpr std::size_t count = 64;
    const cuda_utils::device_buffer<float> buf(count);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), count);
    EXPECT_EQ(buf.bytes(), count * sizeof(float));
    EXPECT_TRUE(static_cast<bool>(buf));
}

TEST(DeviceBuffer, MoveConstruct) {
    cuda_utils::device_buffer<float> src(32);
    const float *original_ptr = src.data();

    const cuda_utils::device_buffer<float> dst(std::move(src));

    EXPECT_EQ(dst.data(), original_ptr);
    EXPECT_EQ(dst.size(), 32u);
    EXPECT_EQ(src.data(), nullptr); // NOLINT(bugprone-use-after-move)
}

TEST(DeviceBuffer, MoveAssignReplaces) {
    cuda_utils::device_buffer<float> src(32);
    const float *original_ptr = src.data();

    cuda_utils::device_buffer<float> dst(16);
    dst = std::move(src);

    EXPECT_EQ(dst.data(), original_ptr);
    EXPECT_EQ(dst.size(), 32u);
}

TEST(DeviceBuffer, Reset) {
    cuda_utils::device_buffer<float> buf(64);
    buf.reset();
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_FALSE(static_cast<bool>(buf));
}

TEST(DeviceBuffer, RoundtripCopy) {
    constexpr std::size_t count = 8;
    const float host_in[count] = {1, 2, 3, 4, 5, 6, 7, 8};
    float host_out[count] = {};

    cuda_utils::device_buffer<float> buf(count);
    cuda_utils::copy_to_device(buf, host_in, count);
    cuda_utils::copy_to_host(host_out, buf, count);

    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(host_out[i], host_in[i]) << "index " << i;
    }
}

// ---- CudaEventTimer ----

TEST(CudaEventTimer, ElapsedNonNegative) {
    cuda_utils::CudaEventTimer timer;
    timer.record_start();
    CUDA_TRY(cudaDeviceSynchronize());
    timer.record_stop();
    EXPECT_GE(timer.elapsed_ms(), 0.0F);
}
