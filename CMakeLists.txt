cmake_minimum_required(VERSION 3.22)

project(cuda_practice LANGUAGES CXX CUDA)

# ---- Language standards ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Options ----
option(ENABLE_WARNINGS "Enable lots of warnings" ON)
option(ENABLE_TIDY "Enable clang-tidy (for C++ sources)" ON)

# ---- clang-tidy ----
if (ENABLE_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if (NOT CLANG_TIDY_EXE)
    message(WARNING "clang-tidy not found; ENABLE_TIDY=ON but clang-tidy is unavailable.")
  endif()
endif()

# ---- Helper: apply common CUDA/CXX settings to a target ----
# Usage: configure_cuda_target(<target>)
# Call this in every chapter's CMakeLists.txt after add_executable/add_library.
function(configure_cuda_target TARGET)
  set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "87;89"  # sm_87: Jetson Orin Nano (Ampere); sm_89: RTX 4090 (Ada Lovelace)
  )
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
  if (ENABLE_WARNINGS)
    if (MSVC)
      target_compile_options(${TARGET} PRIVATE /W4)
    else()
      target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
  endif()
  if (ENABLE_TIDY AND CLANG_TIDY_EXE)
    set_target_properties(${TARGET} PROPERTIES
      CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--warnings-as-errors=*;--quiet"
    )
  endif()
endfunction()

# ---- Dependencies ----
include(FetchContent)
FetchContent_Declare(
  googletest
  URL      https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# ---- Shared utilities ----
add_subdirectory(src/shared)

# ---- Chapters ----
add_subdirectory(src/ch1)
add_subdirectory(tests/ch1)
add_subdirectory(src/ch4)
add_subdirectory(tests/ch4)
