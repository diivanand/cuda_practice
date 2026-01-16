cmake_minimum_required(VERSION 3.22)

project(cuda_practice LANGUAGES CXX CUDA)

# ---- Language standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17) # CUDA toolchains vary; C++20 in device code depends on CUDA version.
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Options ----
option(ENABLE_WARNINGS "Enable lots of warnings" ON)
option(ENABLE_TIDY "Enable clang-tidy (for C++ sources)" ON)

# ---- Build target ----
add_executable(cuda_practice
  src/main.cu
)

target_include_directories(cuda_practice PRIVATE src)

# Good defaults for CUDA compilation
set_target_properties(cuda_practice PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 75  # or: 75;80;86;89 etc
)

# Host compiler flags
if (ENABLE_WARNINGS)
  if (MSVC)
    target_compile_options(cuda_practice PRIVATE /W4)
  else()
    target_compile_options(cuda_practice PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# CUDA flags (nvcc)
target_compile_options(cuda_practice PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ---- clang-tidy integration ----
if (ENABLE_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if (CLANG_TIDY_EXE)
    # Apply to C++ compilation. (CMake won't reliably run clang-tidy on .cu via nvcc.)
    set(CMAKE_CXX_CLANG_TIDY
      ${CLANG_TIDY_EXE};
      --warnings-as-errors=*;
      --quiet
    )
  else()
    message(WARNING "clang-tidy not found; ENABLE_TIDY=ON but clang-tidy is unavailable.")
  endif()
endif()

