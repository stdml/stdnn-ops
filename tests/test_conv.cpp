#include <ttl/nn/ops>
#include <ttl/nn/testing>
#include <ttl/tensor>

TEST(conv_test, test1)
{
    const uint32_t n = 2;
    const uint32_t c = 3;
    const uint32_t h = 128;
    const uint32_t w = 128;

    const uint32_t d = 8;

    const auto x = ttl::tensor<float, 4>(n, h, w, c);
    {
        const uint32_t r = 3;
        const uint32_t s = 3;
        const auto y = ttl::tensor<float, 4>(r, s, c, d);
        {
            const auto conv = nn::ops::conv<nn::ops::nhwc>();
            const auto z = ttl::tensor<float, 4>(conv(x.shape(), y.shape()));
            ASSERT_EQ(z.shape(), ttl::internal::basic_shape<4>(n, 126, 126, d));
            conv(ref(z), view(x), view(y));
        }
        {
            using conv = nn::ops::conv<nn::ops::nhwc>;
            const auto op = conv(conv::padding(1, 1));
            const auto z = ttl::tensor<float, 4>(op(x.shape(), y.shape()));
            ASSERT_EQ(z.shape(), ttl::internal::basic_shape<4>(n, h, w, d));
            op(ref(z), view(x), view(y));
        }
    }
    {
        const auto y = ttl::tensor<float, 4>(4, 4, c, d);
        using conv = nn::ops::conv<nn::ops::nhwc>;
        const auto op = conv(conv::padding(2, 2), conv::stride(4, 4));
        const auto z = ttl::tensor<float, 4>(op(x.shape(), y.shape()));
        ASSERT_EQ(z.shape(), ttl::internal::basic_shape<4>(n, 33, 33, d));
        op(ref(z), view(x), view(y));
    }
}
