#include <ttl/algorithm>

#include <nn/bits/engines/linag.hpp>

#include "testing.hpp"

using la = nn::engines::linag<nn::engines::plain>;

void test_mm(int k, int m, int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(k, m);
    const auto y = ttl::tensor<R, 2>(m, n);
    ttl::fill(ref(x), 2);
    ttl::fill(ref(y), 3);

    const auto z = ttl::tensor<R, 2>(k, n);
    la::mm(view(x), view(y), ref(z));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) { ASSERT_EQ(z.at(i, j), m * 6); }
    }
}

TEST(plain_la_test, test_mm)
{
    test_all_permutations(test_mm, 3, 5, 7);
    test_all_permutations(test_mm, 5, 7, 9);
    test_all_permutations(test_mm, 10, 100, 1000);
}

void test_mmt(int k, int m, int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(k, m);
    const auto y = ttl::tensor<R, 2>(n, m);
    ttl::fill(ref(x), 2);
    ttl::fill(ref(y), 3);

    const auto z = ttl::tensor<R, 2>(k, n);
    la::mmt(view(x), view(y), ref(z));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) { ASSERT_EQ(z.at(i, j), m * 6); }
    }
}

TEST(plain_la_test, test_mmt)
{
    test_all_permutations(test_mmt, 3, 5, 7);
    test_all_permutations(test_mmt, 5, 7, 9);
    test_all_permutations(test_mmt, 10, 100, 1000);
}

void test_mtm(int k, int m, int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(m, k);
    const auto y = ttl::tensor<R, 2>(m, n);
    ttl::fill(ref(x), 2);
    ttl::fill(ref(y), 3);

    const auto z = ttl::tensor<R, 2>(k, n);
    la::mtm(view(x), view(y), ref(z));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) { ASSERT_EQ(z.at(i, j), m * 6); }
    }
}

TEST(plain_la_test, test_mtm)
{
    test_all_permutations(test_mtm, 3, 5, 7);
    test_all_permutations(test_mtm, 5, 7, 9);
    test_all_permutations(test_mtm, 10, 100, 1000);
}

void test_mv(int n, int m)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(n, m);
    const auto y = ttl::tensor<R, 1>(m);
    ttl::fill(ref(x), 2);
    ttl::fill(ref(y), 3);
    const auto z = ttl::tensor<R, 1>(n);
    la::mv(view(x), view(y), ref(z));
    for (int i = 0; i < n; ++i) { ASSERT_EQ(z.at(i), 2 * 3 * m); }
}

TEST(plain_la_test, test_mv) { test_all_permutations(test_mv, 2, 3); }

void test_vm(int n, int m)
{
    using R = int;
    const auto x = ttl::tensor<R, 1>(m);
    const auto y = ttl::tensor<R, 2>(m, n);
    ttl::fill(ref(x), 2);
    ttl::fill(ref(y), 3);
    const auto z = ttl::tensor<R, 1>(n);
    la::vm(view(x), view(y), ref(z));
    for (int i = 0; i < n; ++i) { ASSERT_EQ(z.at(i), 2 * 3 * m); }
}

TEST(plain_la_test, test_vm) { test_all_permutations(test_vm, 2, 3); }
