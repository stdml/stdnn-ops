#include <algorithm>

#include <ttl/algorithm>
#include <ttl/tensor>

#include <nn/ops>
#include <nn/testing>

void test_mm(int k, int m, int n)
{
    using mul = nn::ops::matmul;
    const auto x = ttl::tensor<int, 2>(k, m);
    const auto y = ttl::tensor<int, 2>(m, n);
    ttl::fill(ref(x), 2);
    ttl::fill(ref(y), 3);

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
