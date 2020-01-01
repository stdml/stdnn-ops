#include <ttl/nn/bits/traits/multi_linear_sample.hpp>
#include <ttl/nn/testing>

void test_linear_sample_ksr_nm(int ksize, int stride, int rate, int n, int m)
{
    ttl::nn::traits::linear_sample_trait sample(ksize, stride, rate);

    ASSERT_EQ(sample(n), m);
    ASSERT_EQ(sample(0, 0), 0);
    ASSERT_EQ(sample(m - 1, ksize - 1), n - 1);

    std::vector<int> x(n);
    std::fill(x.begin(), x.end(), 0);
    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < ksize; ++k) { x[sample(j, k)] += 1; }
    }
    ASSERT_EQ(std::accumulate(x.begin(), x.end(), 0), m * ksize);
    ASSERT_EQ(x[0], 1);
    ASSERT_EQ(x[n - 1], 1);
}

void test_linear_sample_ksr_m(int k, int s, int r, int m)
{
    // patch_size = (k - 1) * rate + 1
    // n = (m - 1) * stride + patch_size = (m - 1) * stride + (k - 1) * rate + 1
    int n = (m - 1) * s + (k - 1) * r + 1;
    test_linear_sample_ksr_nm(k, s, r, n, m);
}

void test_linear_sample_1()
{
    /*
    k, s, r = 3, 7, 4
    0123456789ABCDEF0123456789ABCD // n = 30, pad_l = pad_r = 0
    *   *   *                      // ksize = 3, rate = 4, patch_size = 9
           *   *   *               // stride = 7
                  *   *   *
                         *   *   *
                                   // m = (30 - 9) / 7 + 1 = 4
    */
    test_linear_sample_ksr_nm(3, 7, 4, 30, 4);

    for (int k = 1; k <= 3; ++k) {
        for (int s = 1; s <= 3; ++s) {
            for (int r = 1; r <= 3; ++r) {
                for (int m = 1; m <= 3; ++m) {
                    test_linear_sample_ksr_m(k, s, r, m);
                }
            }
        }
    }
}

TEST(linear_sample_test, test_1) { test_linear_sample_1(); }

TEST(linear_sample_test, test_2)
{
    {
        using sample_t = ttl::nn::traits::multi_linear_sample_trait<2, size_t>;

        sample_t sample(sample_t::ksize(1, 1));
        const auto y = sample(ttl::shape<2>(10, 11));
        ASSERT_EQ(y, ttl::shape<2>(10, 11));
    }
    {
        using sample_t = ttl::nn::traits::multi_linear_sample_trait<3, size_t>;

        sample_t sample(sample_t::ksize(1, 2, 3));
        const auto y = sample(ttl::shape<3>(9, 8, 7));
        ASSERT_EQ(y, ttl::shape<3>(9, 7, 5));
    }
}

template <typename dim_t>
void test_valid_padding_ksize_3(dim_t n, dim_t s, dim_t pad_l, dim_t pad_r)
{
    using sample_t = ttl::nn::traits::linear_sample_trait<size_t>;
    const auto padding = sample_t::valid_padding(3, s, 1, n);
    const auto [u, v] = padding.dims();
    ASSERT_EQ(u, pad_l);
    ASSERT_EQ(v, pad_r);
}

template <typename dim_t>
void test_same_padding_ksize_3(dim_t n, dim_t s, dim_t pad_l, dim_t pad_r)
{
    using sample_t = ttl::nn::traits::linear_sample_trait<dim_t>;
    const auto padding = sample_t::same_padding(3, s, 1, n);
    const auto [u, v] = padding.dims();
    ASSERT_EQ(u, pad_l);
    ASSERT_EQ(v, pad_r);
}

TEST(linear_sample_test, test_auto_padding)
{
    test_valid_padding_ksize_3<uint8_t>(56, 1, 0, 0);
    test_valid_padding_ksize_3<uint8_t>(56, 2, 0, 1);

    test_same_padding_ksize_3<uint8_t>(56, 1, 1, 1);
    test_same_padding_ksize_3<uint8_t>(112, 2, 0, 1);
}
