#include <ttl/tensor>

#include <nn/ops>

#include "testing.hpp"

TEST(add_pointwise, test_relu)
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
