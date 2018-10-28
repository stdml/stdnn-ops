#include "testing.hpp"

#include <nn/layers>
#include <stdtensor>

TEST(activation_test, test1)
{
    using relu_op = nn::ops::pointwise<nn::ops::relu>;
    using relu_layer = nn::layers::activation<relu_op>;

    const int k = 10;
    const auto x = ttl::tensor<float, 1>(k);
    const auto y = relu_layer()(ref(x));
}
