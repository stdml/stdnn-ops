#include <ttl/nn/layers>
#include <ttl/nn/testing>
#include <ttl/tensor>

TEST(activation_test, test1)
{
    using relu_layer = ttl::nn::layers::activation<ttl::nn::ops::relu>;

    const int k = 10;
    const auto x = ttl::tensor<float, 1>(k);
    const auto y = relu_layer()(ref(x));
}
