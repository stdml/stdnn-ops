#include <string>

#include <ttl/nn/ops>
#include <ttl/tensor>

#include "utils.hpp"

void example_mlp()
{
    int n = 10;
    int h = 28;
    int w = 28;
    int c = 3;

    using conv_nhwc = ttl::nn::ops::conv<ttl::nn::ops::nhwc>;
    const auto conv = conv_nhwc(conv_nhwc::padding(1, 1));

    using max_pool_nhwc =
        ttl::nn::ops::pool<ttl::nn::ops::pool_max, ttl::nn::ops::nhwc>;
    const auto pool = max_pool_nhwc();

    auto x = ttl::tensor<float, 4>(n, h, w, c);

    auto w1 = ttl::tensor<float, 4>(3, 3, c, 32);
    auto l1 = ttl::tensor<float, 4>(conv(x.shape(), w1.shape()));
    conv(ref(l1), view(x), view(w1));
    show_signature(l1, x, w1);

    auto w2 = ttl::tensor<float, 4>(3, 3, 32, 64);
    auto l2 = ttl::tensor<float, 4>(conv(l1.shape(), w2.shape()));
    conv(ref(l2), view(l1), view(w2));
    show_signature(l2, l1, w2);

    auto l3 = ttl::tensor<float, 4>(pool(l2.shape()));
    pool(ref(l3), view(l2));
    show_signature(l3, l2);

    auto l4 = ttl::tensor<float, 4>(pool(l3.shape()));
    pool(ref(l4), view(l3));
    show_signature(l4, l3);
}

int main()
{
    example_mlp();
    return 0;
}
