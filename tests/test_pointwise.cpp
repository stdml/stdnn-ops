#include <ttl/algorithm>
#include <ttl/tensor>

#include <nn/bits/ops/pointwise.hpp>

#include "testing.hpp"

TEST(pointwise_test, test_relu)
{
    using R = float;
    const int k = 10;
    const auto x = ttl::tensor<R, 1>(k);
    const auto y = ttl::tensor<R, 1>(x.shape());

    using relu = nn::ops::pointwise<nn::ops::relu>;

    for (int i = 0; i < k; ++i) { x.at(i) = i - 4.5; }
    relu()(ref(y), view(x));
    for (int i = 0; i < k; ++i) {
        if (i < 5) {
            ASSERT_FLOAT_EQ(y.at(i), 0);
        } else {
            ASSERT_FLOAT_EQ(y.at(i), i - 4.5);
        }
    }
}

TEST(pointwise_test, test_lambda)
{
    const int n = 100;
    ttl::tensor<uint8_t, 3> x(n, 28, 28);
    ttl::tensor<float, 3> y(x.shape());

    ttl::fill(ref(x), static_cast<uint8_t>(1));

    auto f = [](uint8_t p) { return static_cast<float>(p / 255.0); };
    nn::ops::pointwise<decltype(f)> op(f);
    op(ref(y), view(x));
    ASSERT_EQ(y.data()[0], static_cast<float>(1.0 / 255.0));
}
