#include <cmath>

#include <ttl/algorithm>
#include <ttl/nn/ops>
#include <ttl/nn/testing>
#include <ttl/tensor>

TEST(xentropy_test, test_1)
{
    using R = float;
    const R e = std::exp((R)1.0);

    {
        for (int k = 1; k <= 10; ++k) {
            const auto x = ttl::tensor<R, 1>(k);
            const auto y = ttl::tensor<R, 1>(x.shape());
            ttl::fill(ref(x), (R)1.0);
            ttl::fill(ref(y), e);

            const auto loss = ttl::nn::ops::xentropy();
            const auto z = ttl::tensor<R, 0>(loss(x.shape(), y.shape()));

            loss(ref(z), view(x), view(y));
            ASSERT_FLOAT_EQ(z.data()[0], -k);
        }
    }

    {
        const int n = 2;
        for (int k = 1; k <= 10; ++k) {
            const auto x = ttl::tensor<R, 2>(n, k);
            const auto y = ttl::tensor<R, 2>(x.shape());
            ttl::fill(ref(x), (R)1.0);
            ttl::fill(ref(y), e);

            const auto loss = ttl::nn::ops::xentropy();
            const auto z = ttl::tensor<R, 1>(loss(x.shape(), y.shape()));
            ASSERT_EQ(z.shape(), ttl::shape<1>(2));

            loss(ref(z), view(x), view(y));
            for (int i = 0; i < n; ++i) { ASSERT_FLOAT_EQ(z.at(i), -k); }
        }
    }
}
