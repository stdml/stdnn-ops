
#include <ttl/algorithm>
#include <ttl/nn/bits/ops/gradients/conv2d.hpp>
#include <ttl/nn/testing>

TEST(conv_grad_test, test1)
{
    using image_order = ttl::nn::ops::nhwc;
    using filter_order = ttl::nn::ops::rscd;

    using F = ttl::nn::ops::conv<image_order, filter_order>;
    using G0 = ttl::nn::ops::grad::conv<image_order, filter_order, 0>;
    using G1 = ttl::nn::ops::grad::conv<image_order, filter_order, 1>;

    const F f;
    const G0 g0(f);
    const G1 g1(f);

    const int n = 10;
    const int h = 28;
    const int w = 28;
    const int c = 3;
    const int r = 3;
    const int s = 3;
    const int d = 6;

    ttl::shape<4> x_shape(n, h, w, c);
    ttl::shape<4> y_shape(r, s, c, d);
    ttl::shape<4> z_shape(f(x_shape, y_shape));

    using R = int;

    ttl::tensor<R, 4> x(x_shape);
    ttl::tensor<R, 4> y(y_shape);
    ttl::tensor<R, 4> z(z_shape);

    ttl::fill(ref(x), static_cast<R>(1));
    ttl::fill(ref(y), static_cast<R>(1));
    f(ref(z), view(x), view(y));

    ttl::tensor<R, 4> gx(x_shape);
    ttl::tensor<R, 4> gy(y_shape);
    ttl::tensor<R, 4> gz(z_shape);

    ttl::fill(ref(gz), static_cast<R>(1));
    g0(ref(gx), view(gz), view(z), view(x), view(y));
    g1(ref(gy), view(gz), view(z), view(x), view(y));

    // TODO: ASSERT
}
