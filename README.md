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
./build/debug/cuda_practice
# or
./build/release/cuda_practice
```

Expected output (1M-element vector add, `out[i] = i + 2.0`):

```
out[0]   = 2
out[n-1] = 1.04858e+06
```

## Tests

Tests use [Google Test](https://github.com/google/googletest) (fetched automatically by CMake) and are run through CTest. The test binary is `cuda_practice_tests`.

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

You can also run the test binary directly to get Google Test's native output:

```bash
./build/debug/cuda_practice_tests
```

### Granular test runs

`gtest_discover_tests()` registers each `TEST(Suite, Case)` as its own CTest entry, so `-R` filters on individual cases:

```bash
# All tests in the VecAdd suite
ctest --test-dir build/debug -R "VecAdd"

# One specific test case
ctest --test-dir build/debug -R "VecAdd.SmallKnownInput"

# All DeviceBuffer tests
ctest --test-dir build/debug -R "DeviceBuffer"

# Exclude a suite
ctest --test-dir build/debug -E "VecAdd"

# List registered test names without running
ctest --test-dir build/debug -N

# Re-run only previously failing tests
ctest --test-dir build/debug --rerun-failed

# Stop on first failure
ctest --test-dir build/debug --stop-on-failure

# Parallel execution
ctest --test-dir build/debug -j4
```

Alternatively, pass Google Test's own flags directly to the binary for the same filtering without CTest:

```bash
# Run one suite
./build/debug/cuda_practice_tests --gtest_filter="VecAdd.*"

# Run one case
./build/debug/cuda_practice_tests --gtest_filter="VecAdd.SmallKnownInput"

# Exclude a suite
./build/debug/cuda_practice_tests --gtest_filter="-DeviceBuffer.*"

# List all test cases without running
./build/debug/cuda_practice_tests --gtest_list_tests
```

Test output is written to `build/debug/Testing/Temporary/LastTest.log`.

## Static analysis

clang-tidy runs automatically during the build when `ENABLE_TIDY=ON` (the default for both presets). Rules are defined in `.clang-tidy`. All warnings are treated as errors.

To disable tidy for a one-off build:

```bash
cmake --preset ninja-clang-debug -DENABLE_TIDY=OFF
cmake --build --preset debug
```

## Project layout

```
src/
  main.cu          # entry point, launches vec_add kernel
  vec_add.cuh      # __global__ vector addition kernel
  cuda_utils.cuh   # RAII device_buffer, CUDA_TRY error macro
tests/
  test_vec_add.cu  # Google Test cases for vec_add kernel and device_buffer
CMakeLists.txt
CMakePresets.json
.clang-tidy
```
