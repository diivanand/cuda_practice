add_executable(ch4 main.cu
        check_prime_gpu.cuh
        check_prime_cpu.h
)
target_include_directories(ch4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
configure_cuda_target(ch4)
