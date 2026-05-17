add_executable(ch4_tests test_check_prime_gpu.cu test_check_prime_cpu.cpp)
target_link_libraries(ch4_tests PRIVATE ch4_lib GTest::gtest_main)
configure_cuda_target(ch4_tests)
gtest_discover_tests(ch4_tests TEST_PREFIX "ch4/")
