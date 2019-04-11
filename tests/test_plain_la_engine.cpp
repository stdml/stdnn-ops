#include <nn/bits/engines/linag.hpp>

#include "testing.hpp"

using la = nn::engines::linag<nn::engines::plain>;

void test_mm(int k, int m, int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(k, m);
    const auto y = ttl::tensor<R, 2>(m, n);
    fill(x, 2);
    fill(y, 3);

    const auto z = ttl::tensor<R, 2>(k, n);
    la::mm(view(x), view(y), ref(z));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) { ASSERT_EQ(z.at(i, j), m * 6); }
    }
}

void test_mm_all(int k, int m, int n)
{
    std::array<int, 3> a({k, m, n});
    do {
        const auto [k, m, n] = a;
        test_mm(k, m, n);
    } while (std::next_permutation(a.begin(), a.end()));
}

TEST(plain_la_test, test_mm)
{
    test_mm_all(3, 5, 7);
    test_mm_all(5, 7, 9);
    test_mm_all(10, 100, 1000);
}

void test_mmt(int k, int m, int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(k, m);
    const auto y = ttl::tensor<R, 2>(n, m);
    fill(x, 2);
    fill(y, 3);

    const auto z = ttl::tensor<R, 2>(k, n);
    la::mmt(view(x), view(y), ref(z));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) { ASSERT_EQ(z.at(i, j), m * 6); }
    }
}

void test_mmt_all(int k, int m, int n)
{
    std::array<int, 3> a({k, m, n});
    do {
        const auto [k, m, n] = a;
        test_mmt(k, m, n);
    } while (std::next_permutation(a.begin(), a.end()));
}

TEST(plain_la_test, test_mmt)
{
    test_mmt_all(3, 5, 7);
    test_mmt_all(5, 7, 9);
    test_mmt_all(10, 100, 1000);
}

void test_mtm(int k, int m, int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(m, k);
    const auto y = ttl::tensor<R, 2>(m, n);
    fill(x, 2);
    fill(y, 3);

    const auto z = ttl::tensor<R, 2>(k, n);
    la::mtm(view(x), view(y), ref(z));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) { ASSERT_EQ(z.at(i, j), m * 6); }
    }
}

void test_mtm_all(int k, int m, int n)
{
    std::array<int, 3> a({k, m, n});
    do {
        const auto [k, m, n] = a;
        test_mtm(k, m, n);
    } while (std::next_permutation(a.begin(), a.end()));
}

TEST(plain_la_test, test_mtm)
{
    test_mtm_all(3, 5, 7);
    test_mtm_all(5, 7, 9);
    test_mtm_all(10, 100, 1000);
}

void test_mv(int n, int m)
{
    using R = int;
    const auto x = ttl::tensor<R, 2>(n, m);
    const auto y = ttl::tensor<R, 1>(m);
    fill(x, 2);
    fill(y, 3);
    const auto z = ttl::tensor<R, 1>(n);
    la::mv(view(x), view(y), ref(z));
    for (int i = 0; i < n; ++i) { ASSERT_EQ(z.at(i), 2 * 3 * m); }
}

TEST(plain_la_test, test_mv)
{
    test_mv(2, 3);
    test_mv(3, 2);
}

void test_vm(int n, int m)
{
    using R = int;
    const auto x = ttl::tensor<R, 1>(m);
    const auto y = ttl::tensor<R, 2>(m, n);
    fill(x, 2);
    fill(y, 3);
    const auto z = ttl::tensor<R, 1>(n);
    la::vm(view(x), view(y), ref(z));
    for (int i = 0; i < n; ++i) { ASSERT_EQ(z.at(i), 2 * 3 * m); }
}

TEST(plain_la_test, test_vm)
{
    test_vm(2, 3);
    test_vm(3, 2);
}

void test_vv(int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 1>(n);
    const auto y = ttl::tensor<R, 1>(n);
    fill(x, 1);
    fill(y, 2);
    const auto z = ttl::tensor<R, 1>(n);
    la::vv(view(x), view(y), ref(z));
    for (int i = 0; i < n; ++i) { ASSERT_EQ(z.at(i), 3); }
}

TEST(plain_la_test, test_vv)
{
    for (int i = 1; i <= 100; ++i) { test_vv(i); }
}
