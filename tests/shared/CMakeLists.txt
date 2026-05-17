add_executable(shared_tests test_cuda_utils.cu)
target_link_libraries(shared_tests PRIVATE shared GTest::gtest_main)
configure_cuda_target(shared_tests)
gtest_discover_tests(shared_tests TEST_PREFIX "shared/")
