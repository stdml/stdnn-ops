#include <cmath>

#include "testing.hpp"

#include <nn/ops>
#include <stdtensor>

TEST(xentropy_test, test_1)
{
    using R = float;
    const R e = std::exp((R)1.0);

    {
        for (int k = 1; k <= 10; ++k) {
            const auto x = ttl::tensor<R, 1>(k);
            const auto y = ttl::tensor<R, 1>(x.shape());
            fill(x, (R)1.0);
            fill(y, e);

            const auto loss = nn::ops::xentropy<1>();
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
            fill(x, (R)1.0);
            fill(y, e);

            const auto loss = nn::ops::xentropy<2>();
            const auto z = ttl::tensor<R, 1>(loss(x.shape(), y.shape()));

            loss(ref(z), view(x), view(y));
            for (int i = 0; i < n; ++i) { ASSERT_FLOAT_EQ(z.at(i), -k); }
        }
    }
}
