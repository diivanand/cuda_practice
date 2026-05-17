add_executable(ch1 main.cu)
target_include_directories(ch1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
configure_cuda_target(ch1)
