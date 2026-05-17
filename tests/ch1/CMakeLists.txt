add_executable(ch1_tests test_vec_add.cu)
target_link_libraries(ch1_tests PRIVATE ch1_lib GTest::gtest_main)
configure_cuda_target(ch1_tests)
gtest_discover_tests(ch1_tests TEST_PREFIX "ch1/")
