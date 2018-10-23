#include <gtest/gtest.h>

#include <nn/ops>
#include <stdtensor>

TEST(pool_test, test1)
{
    const uint32_t n = 10;
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
        {
            const auto op = max_pool_nchw();
            const auto y = ttl::tensor<float, 4>(op(x.shape()));
            ASSERT_EQ(y.shape(), ttl::internal::basic_shape<4>(n, c, 64, 128));
            op(ref(y), view(x));
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
