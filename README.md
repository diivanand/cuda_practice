# cuda_practice

CUDA C++ practice project: vector addition and RAII GPU utilities.

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| CMake | 3.22 | `cmake --version` |
| Ninja | any | `ninja --version` |
| CUDA Toolkit | 11.x+ | `nvcc --version` |
| clang++ | 13+ | host compiler |
| clang-tidy | same as clang++ | optional; skipped if not found |

The presets target **sm_75** (Turing). Change `CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU (e.g. `86` for Ampere, `89` for Ada).

## Configure

Two presets are provided — debug and release — both using Ninja and clang++ as the host compiler.

```bash
# Debug (default for development)
cmake --preset ninja-clang-debug

# Release (optimised)
cmake --preset ninja-clang-release
```

Configured build trees land in `build/debug` and `build/release` respectively. Both emit `compile_commands.json` for clang-tidy and editor tooling.

## Build

```bash
# Debug
cmake --build --preset debug

# Release
cmake --build --preset release

# Parallel build (explicit job count)
cmake --build --preset debug -- -j$(nproc)
```

## Run

```bash
./build/debug/src/ch1/ch1
# or
./build/release/src/ch1/ch1
```

Expected output (1M-element vector add, `out[i] = i + 2.0`):

```
out[0]   = 2
out[n-1] = 1.04858e+06
```

## Tests

Tests use [Google Test](https://github.com/google/googletest) (fetched automatically by CMake) and are run through CTest. Each chapter has its own test binary (`ch1_tests`, `ch4_tests`, …).

```bash
# Build everything including tests, then run all tests
cmake --build --preset debug && ctest --test-dir build/debug

# Verbose output (show stdout/stderr for every test)
ctest --test-dir build/debug -V

# Extra verbose (show cmake/ctest internals too)
ctest --test-dir build/debug -VV

# Output only on failure (quiet otherwise)
ctest --test-dir build/debug --output-on-failure
```

You can also run a chapter's test binary directly to get Google Test's native output:

```bash
./build/debug/tests/ch1/ch1_tests
```

### Granular test runs

`gtest_discover_tests()` registers each `TEST(Suite, Case)` as its own CTest entry prefixed with the chapter name (e.g. `ch1/VecAdd.SmallKnownInput`), so you can filter by chapter, suite, or individual case:

```bash
# All tests for a chapter
ctest --test-dir build/debug -R "^ch1/"

# All tests in a suite across all chapters
ctest --test-dir build/debug -R "VecAdd"

# One specific test case
ctest --test-dir build/debug -R "ch1/VecAdd.SmallKnownInput"

# Exclude a chapter
ctest --test-dir build/debug -E "^ch1/"

# List registered test names without running
ctest --test-dir build/debug -N

# Re-run only previously failing tests
ctest --test-dir build/debug --rerun-failed

# Stop on first failure
ctest --test-dir build/debug --stop-on-failure

# Parallel execution
ctest --test-dir build/debug -j4
```

Alternatively, pass Google Test's own flags directly to a chapter's binary:

```bash
# Run one suite
./build/debug/tests/ch1/ch1_tests --gtest_filter="VecAdd.*"

# Run one case
./build/debug/tests/ch1/ch1_tests --gtest_filter="VecAdd.SmallKnownInput"

# Exclude a suite
./build/debug/tests/ch1/ch1_tests --gtest_filter="-DeviceBuffer.*"

# List all test cases without running
./build/debug/tests/ch1/ch1_tests --gtest_list_tests
```

Test output is written to `build/debug/Testing/Temporary/LastTest.log`.

## Static analysis

clang-tidy is wired up via `CXX_CLANG_TIDY` and runs automatically on any `.cpp`/`.cxx` sources during the build when `ENABLE_TIDY=ON` (the default for both presets). Rules are defined in `.clang-tidy`; all warnings are treated as errors. `.cu` files are compiled by nvcc, which does not integrate with clang-tidy, so it has no effect on them.

To disable tidy for a one-off build:

```bash
cmake --preset ninja-clang-debug -DENABLE_TIDY=OFF
cmake --build --preset debug
```

## Namespaces

Each chapter's public API lives in its own namespace. Shared CUDA utilities are under `cuda_utils`.

| Namespace | Provided by | Contents |
|-----------|-------------|----------|
| `ch1` | `vec_add.cuh` | `vec_add` kernel |
| `ch4` | `check_prime_cpu.h`, `check_prime_gpu.cuh` | `check_prime_cpu`, `check_prime_gpu_kernel` |
| `cuda_utils` | `shared/cuda_utils.cuh` | `device_buffer<T>`, `copy_to_device`, `copy_to_host`, `cuda_fail` |

`CUDA_TRY(expr)` is a macro (macros cannot be namespaced) and is defined globally in `cuda_utils.cuh`.

CUDA `__global__` kernels fully support C++ namespaces — launch them with the qualified name:

```cpp
ch1::vec_add<<<grid, block>>>(a, b, out, n);
```

## Project layout

```
src/
  shared/
    cuda_utils.cuh      # cuda_utils:: — device_buffer<T>, CUDA_TRY, copy helpers
    CMakeLists.txt      # defines shared (INTERFACE); chapters link against it
  ch1/
    main.cu             # entry point, launches ch1::vec_add kernel
    vec_add.cuh         # declaration of ch1::vec_add
    vec_add.cu          # definition of ch1::vec_add
    CMakeLists.txt      # defines ch1_lib (STATIC) and ch1 executable
  ch4/
    main.cu             # entry point (placeholder)
    check_prime_cpu.h   # declaration of ch4::check_prime_cpu
    check_prime_cpu.cpp # definition of ch4::check_prime_cpu
    check_prime_gpu.cuh # declaration of ch4::check_prime_gpu_kernel
    check_prime_gpu.cu  # definition of ch4::check_prime_gpu_kernel
    CMakeLists.txt      # defines ch4_lib (STATIC) and ch4 executable
tests/
  ch1/
    test_vec_add.cu     # Google Test cases for ch1::vec_add and cuda_utils::device_buffer
    CMakeLists.txt      # links ch1_lib and GTest
  ch4/
    test_ch4.cu         # Google Test cases for ch4 (placeholder)
    CMakeLists.txt      # links ch4_lib and GTest
CMakeLists.txt          # root: common settings, dependencies, chapter subdirs
CMakePresets.json
.clang-tidy
```

### Adding a new chapter

1. Create `src/chN/` with your headers, implementation files, `main.cu`, and a `CMakeLists.txt`:
   ```cmake
   add_library(chN_lib STATIC foo.cu bar.cpp foo.cuh bar.h)
   target_include_directories(chN_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
   configure_cuda_target(chN_lib)

   add_executable(chN main.cu)
   target_link_libraries(chN PRIVATE chN_lib)
   configure_cuda_target(chN)
   ```
2. Create `tests/chN/` with your tests and a `CMakeLists.txt`:
   ```cmake
   add_executable(chN_tests test_foo.cu)
   target_link_libraries(chN_tests PRIVATE chN_lib GTest::gtest_main)
   configure_cuda_target(chN_tests)
   gtest_discover_tests(chN_tests TEST_PREFIX "chN/")
   ```
3. Add both to the root `CMakeLists.txt`:
   ```cmake
   add_subdirectory(src/chN)
   add_subdirectory(tests/chN)
   ```
4. Put all chapter-specific symbols in a `chN` namespace. Shared CUDA utilities go in `cuda_utils` (see `src/ch1/cuda_utils.cuh`).
