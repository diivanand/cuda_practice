#include <gtest/gtest.h>

#include "check_prime_cpu.h"

// ---- check_prime_cpu ----

TEST(CheckPrimeCpu, BelowTwo) {
    EXPECT_FALSE(ch4::check_prime_cpu(1));
    EXPECT_FALSE(ch4::check_prime_cpu(0));
    EXPECT_FALSE(ch4::check_prime_cpu(-1));
    EXPECT_FALSE(ch4::check_prime_cpu(-7));
}

TEST(CheckPrimeCpu, Two) { EXPECT_TRUE(ch4::check_prime_cpu(2)); }

TEST(CheckPrimeCpu, EvenComposites) {
    EXPECT_FALSE(ch4::check_prime_cpu(4));
    EXPECT_FALSE(ch4::check_prime_cpu(100));
    EXPECT_FALSE(ch4::check_prime_cpu(1024));
}

TEST(CheckPrimeCpu, OddComposites) {
    EXPECT_FALSE(ch4::check_prime_cpu(9));
    EXPECT_FALSE(ch4::check_prime_cpu(15));
    EXPECT_FALSE(ch4::check_prime_cpu(49));
    EXPECT_FALSE(ch4::check_prime_cpu(561)); // Carmichael number
}

TEST(CheckPrimeCpu, SmallPrimes) {
    for (long long prime : {3LL, 5LL, 7LL, 11LL, 13LL, 17LL, 19LL, 23LL}) {
        EXPECT_TRUE(ch4::check_prime_cpu(prime)) << "expected prime: " << prime;
    }
}

TEST(CheckPrimeCpu, LargePrime) {
    EXPECT_TRUE(ch4::check_prime_cpu(999999937LL));
}

TEST(CheckPrimeCpu, LargeComposite) {
    EXPECT_FALSE(ch4::check_prime_cpu(999999938LL));
}
