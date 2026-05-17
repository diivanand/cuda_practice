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

Tests use [Google Test](https://github.com/google/googletest) (fetched automatically by CMake) and are run through CTest. Each chapter has its own test binary (`ch1_tests`, `ch2_tests`, …).

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

## Project layout

```
src/
  ch1/
    main.cu          # entry point, launches vec_add kernel
    vec_add.cuh      # __global__ vector addition kernel
    cuda_utils.cuh   # RAII device_buffer, CUDA_TRY error macro
    CMakeLists.txt   # defines the `ch1` executable
tests/
  ch1/
    test_vec_add.cu  # Google Test cases for vec_add kernel and device_buffer
    CMakeLists.txt   # defines the `ch1_tests` binary
CMakeLists.txt       # root: common settings, dependencies, chapter subdirs
CMakePresets.json
.clang-tidy
```

### Adding a new chapter

1. Create `src/chapterN/` with your sources and a `CMakeLists.txt`:
   ```cmake
   add_executable(chapterN main.cu)
   target_include_directories(chapterN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
   configure_cuda_target(chapterN)
   ```
2. Create `tests/chapterN/` with your tests and a `CMakeLists.txt`:
   ```cmake
   add_executable(chapterN_tests test_foo.cu)
   target_include_directories(chapterN_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/chapterN)
   target_link_libraries(chapterN_tests PRIVATE GTest::gtest_main)
   configure_cuda_target(chapterN_tests)
   gtest_discover_tests(chapterN_tests TEST_PREFIX "chapterN/")
   ```
3. Add both to the root `CMakeLists.txt`:
   ```cmake
   add_subdirectory(src/chapterN)
   add_subdirectory(tests/chapterN)
   ```
