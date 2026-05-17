add_library(ch1_lib STATIC vec_add.cu vec_add.cuh)
target_include_directories(ch1_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ch1_lib PUBLIC shared)
configure_cuda_target(ch1_lib)

add_executable(ch1 main.cu)
target_link_libraries(ch1 PRIVATE ch1_lib)
configure_cuda_target(ch1)
