#pragma once

namespace ch4 {

// Returns true if num is prime, false otherwise.
// Intended as a CPU reference implementation for validating GPU results.
auto check_prime_cpu(long long num) -> bool;

} // namespace ch4
