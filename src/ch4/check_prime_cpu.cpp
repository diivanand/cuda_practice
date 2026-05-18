#include "check_prime_cpu.h"

#include <iostream>

namespace ch4 {

auto check_prime_cpu(long long num) -> bool {
    if (num < 2) {
        return false;
    }
    if (num == 2) {
        return true;
    }
    if (num % 2 == 0) {
        return false;
    }
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }

    return true;
}

} // namespace ch4
