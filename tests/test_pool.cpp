#include <ttl/nn/bits/ops/pool.hpp>
#include <ttl/nn/testing>

TEST(pool_test, test_hw)
{
    const auto x = ttl::tensor<int, 2>(4, 4);
    const auto y = ttl::tensor<int, 2>(2, 2);

    for (auto i : ttl::range(16)) { x.data()[i] = i; }

    using pool =
        ttl::nn::ops::pool<ttl::nn::traits::pool_max, ttl::nn::traits::hw>;
    pool()(ttl::ref(y), ttl::view(x));

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
        for (auto k : ttl::range(n)) {
            for (auto i : ttl::range(4)) {
                for (auto j : ttl::range(4)) { x.at(i, j, k) = cnt++; }
            }
        }
    }

    using pool =
        ttl::nn::ops::pool<ttl::nn::traits::pool_max, ttl::nn::traits::hwc>;
    pool()(ttl::ref(y), ttl::view(x));

    for (auto i : ttl::range(n)) {
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
        using max_pool_nhwc = ttl::nn::ops::pool<ttl::nn::traits::pool_max,
                                                 ttl::nn::traits::nhwc>;
        const auto x = ttl::tensor<float, 4>(n, h, w, c);
        {
            const auto op = max_pool_nhwc();
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, 64, 128, c));
            op(ttl::ref(y), ttl::view(x));
        }
        {
            const auto op = max_pool_nhwc(max_pool_nhwc::ksize(4, 4));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, 32, 64, c));
            op(ttl::ref(y), ttl::view(x));
        }
        {
            const auto op = max_pool_nhwc(max_pool_nhwc::ksize(4, 4),
                                          max_pool_nhwc::stride(2, 2));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, 63, 127, c));
            op(ttl::ref(y), ttl::view(x));
        }
    }

    {
        using max_pool_nchw =
            ttl::nn::ops::pool<ttl::nn::ops::pool_max, ttl::nn::ops::nchw>;
        const auto x = ttl::tensor<float, 4>(n, c, h, w);
        for (auto l : ttl::range(n)) {
            for (auto k : ttl::range(c)) {
                for (auto i : ttl::range(h)) {
                    for (auto j : ttl::range(w)) {
                        x.at(l, k, i, j) =
                            (l * 10 + k) * 1000 * 1000 + i * 1000 + j;
                    }
                }
            }
        }
        {
            const auto op = max_pool_nchw();
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, c, 64, 128));
            op(ttl::ref(y), ttl::view(x));
            for (auto l : ttl::range(n)) {
                for (auto k : ttl::range(c)) {
                    for (auto i : ttl::range(64)) {
                        for (auto j : ttl::range(128)) {
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
            ASSERT_EQ(y.shape(), ttl::make_shape(n, c, 32, 64));
            op(ttl::ref(y), ttl::view(x));
        }
        {
            const auto op = max_pool_nchw(max_pool_nchw::ksize(4, 4),
                                          max_pool_nchw::stride(2, 2));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, c, 63, 127));
            op(ttl::ref(y), ttl::view(x));
        }
    }
}

TEST(pool_test, test_padding)
{
    const uint32_t n = 7;
    const uint32_t h = 128;
    const uint32_t w = 256;
    const uint32_t c = 3;

    {
        using pool =
            ttl::nn::ops::pool<ttl::nn::ops::pool_max, ttl::nn::ops::nhwc>;
        const auto x = ttl::tensor<float, 4>(n, h, w, c);
        {
            const auto op = pool(pool::ksize(3, 3), pool::padding(2, 1));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, 44, 86, c));
            op(ttl::ref(y), ttl::view(x));
        }
        {
            const auto op = pool(pool::ksize(3, 3), pool::padding(1, 1),
                                 pool::stride(1, 1));
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::make_shape(n, 128, 256, c));
            op(ttl::ref(y), ttl::view(x));
        }
    }
}

TEST(pool_test, test_mean)
{
    using pool = ttl::nn::ops::pool<ttl::nn::ops::pool_mean, ttl::nn::ops::hw>;
    const pool op;
    const auto x = ttl::tensor<int, 2>(4, 4);
    std::iota(x.data(), x.data() + 16, 0);
    using add = ttl::nn::ops::add;
    add()(ttl::ref(x), ttl::view(x), ttl::view(x));

    const auto y = ttl::tensor<int, 2>(op(x.shape()));
    op(ttl::ref(y), ttl::view(x));
    ASSERT_EQ(y.data()[0], 5);
    ASSERT_EQ(y.data()[1], 9);
    ASSERT_EQ(y.data()[2], 21);
    ASSERT_EQ(y.data()[3], 25);
}
