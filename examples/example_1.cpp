#include <nn/ops>
#include <stdtensor>

struct examples {
    const uint32_t n = 10;
    const uint32_t c = 3;
    const uint32_t h = 512;
    const uint32_t w = 512;

    void example_add()
    {
        const ttl::internal::basic_shape<4> shape(n, h, w, c);

        const auto x = ttl::tensor<float, 4>(shape);
        const auto y = ttl::tensor<float, 4>(shape);

        const auto op = nn::ops::add();
        const auto z = ttl::tensor<float, 4>(op(x.shape(), y.shape()));

        op(ref(z), view(x), view(y));  // FIXME: leaked from namespace
    }

    void example_conv()
    {
        const uint32_t r = 3;
        const uint32_t s = 3;
        const uint32_t d = 32;

        const auto x = ttl::tensor<float, 4>(n, h, w, c);
        const auto y = ttl::tensor<float, 4>(r, s, c, d);

        {
            const auto conv = nn::ops::conv<nn::ops::nhwc>();
            const auto z = ttl::tensor<float, 4>(conv(x.shape(), y.shape()));
            conv(ref(z), view(x), view(y));
        }
        {
            using conv = nn::ops::conv<nn::ops::nhwc>;
            const auto op = conv(conv::padding(1, 1));
            const auto z = ttl::tensor<float, 4>(op(x.shape(), y.shape()));
            op(ref(z), view(x), view(y));
        }
        {
            const auto y = ttl::tensor<float, 4>(4, 4, c, d);
            using conv = nn::ops::conv<nn::ops::nhwc>;
            const auto op = conv(conv::padding(2, 2), conv::stride(4, 4));
            const auto z = ttl::tensor<float, 4>(op(x.shape(), y.shape()));
            op(ref(z), view(x), view(y));
        }
    }

    void example_pool()
    {
        const auto x = ttl::tensor<float, 4>(n, h, w, c);
        using max_pool_nhwc = nn::ops::pool<nn::ops::pool_max, nn::ops::nhwc>;
        const auto op = max_pool_nhwc();
        const auto y = ttl::tensor<float, 4>(op(x.shape()));
        op(ref(y), view(x));
    }
};

int main()
{
    examples e;
    e.example_add();
    e.example_conv();
    e.example_pool();

    // should be done in cg
    // using f = nn::ops::add;
    // f(x, y)(z);
    // f.in(x, y).out(z);
    // f.out(x, y).in(z);

    // z << f << (x, y);
    return 0;
}
