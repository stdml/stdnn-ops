#include <string>

#include <ttl/nn/layers>
#include <ttl/tensor>

#include "utils.hpp"

void example_mlp()
{
    int n = 10;
    int h = 32;
    int w = 32;
    int c = 3;

    using conv_nhwc = ttl::nn::layers::conv<ttl::nn::ops::nhwc>;

    using max_pool_nhwc =
        ttl::nn::layers::pool<ttl::nn::ops::pool_max, ttl::nn::ops::nhwc>;
    const auto pool = max_pool_nhwc(max_pool_nhwc::ksize(4, 4));

    using dense = ttl::nn::layers::dense<>;

    auto x = ttl::tensor<float, 4>(n, h, w, c);
    auto l1 = conv_nhwc(32, conv_nhwc::ksize(3, 3))(ref(x));
    show_signature(*l1, x, l1.arg<0>(), l1.arg<1>());

    auto l2 = conv_nhwc(64, conv_nhwc::ksize(3, 3))(ref(*l1));
    show_signature(*l2, *l1, l2.arg<0>(), l2.arg<1>());

    auto l3 = pool(ref(*l2));
    show_signature(*l3, *l2);

    auto [d0, d1, d2, d3] = (*l3).shape().dims();
    ttl::tensor_ref<float, 2> l3_flat((*l3).data(), d0, d1 * d2 * d3);
    show_signature(l3_flat, *l3);

    auto l4 = dense(1000)(l3_flat);
    show_signature(*l4, l3_flat, l4.arg<0>(), l4.arg<1>());

    using softmax = ttl::nn::layers::activation<ttl::nn::ops::softmax>;
    auto l5 = softmax()(ref(*l4));
    show_signature(*l5, *l4);

    auto y_ = ttl::tensor<float, 2>((*l5).shape());
    const auto loss = ttl::nn::ops::xentropy();

    auto l = ttl::tensor<float, 1>(loss((*l5).shape(), y_.shape()));
    show_signature(l, *l4, y_);
}

int main()
{
    example_mlp();
    return 0;
}
