add_library(shared INTERFACE cuda_utils.cuh)
target_include_directories(shared INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
