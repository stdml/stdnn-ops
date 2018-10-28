#include "testing.hpp"

#include <nn/ops>
#include <stdtensor>

TEST(pool_test, test_hw)
{
    const auto x = ttl::tensor<int, 2>(4, 4);
    const auto y = ttl::tensor<int, 2>(2, 2);

    for (auto i : range(16)) { x.data()[i] = i; }

    using pool = nn::ops::pool<nn::ops::pool_max, nn::ops::hw>;
    pool()(ref(y), view(x));

    ASSERT_EQ(y.at(0, 0), 5);
    ASSERT_EQ(y.at(0, 1), 7);
    ASSERT_EQ(y.at(1, 0), 13);
    ASSERT_EQ(y.at(1, 1), 15);
}

TEST(pool_test, test_hwc)
{
    const int n = 6;
    const auto x = ttl::tensor<int, 3>(4, 4, n);
    const auto y = ttl::tensor<int, 3>(2, 2, n);

    {
        int cnt = 0;
        for (auto k : range(n)) {
            for (auto i : range(4)) {
                for (auto j : range(4)) { x.at(i, j, k) = cnt++; }
            }
        }
    }

    using pool = nn::ops::pool<nn::ops::pool_max, nn::ops::hwc>;
    pool()(ref(y), view(x));

    for (auto i : range(n)) {
        ASSERT_EQ(y.at(0, 0, i), 5 + 16 * i);
        ASSERT_EQ(y.at(0, 1, i), 7 + 16 * i);
        ASSERT_EQ(y.at(1, 0, i), 13 + 16 * i);
        ASSERT_EQ(y.at(1, 1, i), 15 + 16 * i);
    }
}

TEST(pool_test, test_4d)
{
    const uint32_t n = 7;
    const uint32_t c = 3;
    const uint32_t h = 128;
    const uint32_t w = 256;

    {
        using max_pool_nhwc = nn::ops::pool<nn::ops::pool_max, nn::ops::nhwc>;
        const auto x = ttl::tensor<float, 4>(n, h, w, c);
        {
            const auto op = max_pool_nhwc();
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, 64, 128, c));
            op(ref(y), view(x));
        }
        {
            const auto op = max_pool_nhwc(max_pool_nhwc::ksize(4, 4));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, 32, 64, c));
            op(ref(y), view(x));
        }
        {
            const auto op = max_pool_nhwc(max_pool_nhwc::ksize(4, 4),
                                          max_pool_nhwc::stride(2, 2));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, 63, 127, c));
            op(ref(y), view(x));
        }
    }

    {
        using max_pool_nchw = nn::ops::pool<nn::ops::pool_max, nn::ops::nchw>;
        const auto x = ttl::tensor<float, 4>(n, c, h, w);
        for (auto l : range(n)) {
            for (auto k : range(c)) {
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        x.at(l, k, i, j) =
                            (l * 10 + k) * 1000 * 1000 + i * 1000 + j;
                    }
                }
            }
        }
        {
            const auto op = max_pool_nchw();
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, c, 64, 128));
            op(ref(y), view(x));
            for (auto l : range(n)) {
                for (auto k : range(c)) {
                    for (auto i : range(64)) {
                        for (auto j : range(128)) {
                            const float val = (l * 10 + k) * 1000 * 1000 +
                                              (i * 2 + 1) * 1000 + (j * 2 + 1);
                            ASSERT_EQ(y.at(l, k, i, j), val);
                        }
                    }
                }
            }
        }
        {
            const auto op = max_pool_nchw(max_pool_nchw::ksize(4, 4));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, c, 32, 64));
            op(ref(y), view(x));
        }
        {
            const auto op = max_pool_nchw(max_pool_nchw::ksize(4, 4),
                                          max_pool_nchw::stride(2, 2));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, c, 63, 127));
            op(ref(y), view(x));
        }
    }
}
