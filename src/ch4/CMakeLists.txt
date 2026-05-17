add_library(ch4_lib STATIC
    check_prime_gpu.cu
    check_prime_cpu.cpp
    check_prime_gpu.cuh
    check_prime_cpu.h
)
target_include_directories(ch4_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
configure_cuda_target(ch4_lib)

add_executable(ch4 main.cu)
target_link_libraries(ch4 PRIVATE ch4_lib)
configure_cuda_target(ch4)
