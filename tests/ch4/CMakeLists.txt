add_executable(ch4_tests test_ch4.cu)
target_link_libraries(ch4_tests PRIVATE ch4_lib GTest::gtest_main)
configure_cuda_target(ch4_tests)
gtest_discover_tests(ch4_tests TEST_PREFIX "ch4/")
