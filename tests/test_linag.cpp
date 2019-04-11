#include <algorithm>

#include <stdtensor>

#include <nn/ops>

#include "testing.hpp"

void test_mm(int k, int m, int n)
{
    using mul = nn::ops::matmul;
    const auto x = ttl::tensor<int, 2>(k, m);
    const auto y = ttl::tensor<int, 2>(m, n);
    fill(x, 2);
    fill(y, 3);

    const auto op = mul();
    const auto z = ttl::tensor<int, 2>(op(x.shape(), y.shape()));
    ASSERT_EQ(z.shape(), ttl::internal::basic_shape<2>(k, n));

    op(ref(z), view(x), view(y));
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

TEST(matmul_test, test1)
{
    test_mm_all(3, 5, 7);
    test_mm_all(5, 7, 9);
    test_mm_all(10, 100, 1000);
}

void test_vv(int n)
{
    using R = int;
    const auto x = ttl::tensor<R, 1>(n);
    const auto y = ttl::tensor<R, 1>(n);
    fill(x, 1);
    fill(y, 2);
    const auto z = ttl::tensor<R, 1>(n);
    nn::engines::linag<nn::engines::default_engine>::vv(view(x), view(y),
                                                        ref(z));
    for (int i = 0; i < n; ++i) { ASSERT_EQ(z.at(i), 3); }
}

TEST(linag_test, test_vv)
{
    for (int i = 1; i <= 100; ++i) { test_vv(i); }
}
