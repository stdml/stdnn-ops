#include <nn/experimental/bits/ops/grad/bias.hpp>
#include <nn/ops>

#include "testing.hpp"

template <typename R, typename F, typename G0, typename G1, ttl::rank_t r0,
          ttl::rank_t r1>
void test_binary_grad(const F &f, const G0 &g0, const G1 &g1,
                      const ttl::shape<r0> &x_shape,
                      const ttl::shape<r1> &y_shape)
{
    ttl::tensor<R, r0> x(x_shape);
    ttl::tensor<R, r1> y(y_shape);

    const auto z_shape = f(x_shape, y_shape);
    ttl::tensor<R, decltype(z_shape)::rank> z(z_shape);

    ttl::tensor<R, r0> gx(x.shape());
    ttl::tensor<R, r1> gy(y.shape());
    ttl::tensor<R, decltype(z_shape)::rank> gz(z.shape());

    f(ref(z), view(x), view(y));

    g0(ref(gx), view(gz), view(z), view(x), view(y));
    g1(ref(gy), view(gz), view(z), view(x), view(y));
}

TEST(linear_bias_test, test_1)
{
    using R = int32_t;

    using F = nn::ops::add_bias<nn::ops::hw>;
    using G0 = nn::experimental::ops::grad::add_bias<nn::ops::hw, 0>;
    using G1 = nn::experimental::ops::grad::add_bias<nn::ops::hw, 1>;
    G0 g0;
    G1 g1;
    F f;

    const int n = 10;
    const int l = 32;
    test_binary_grad<R>(f, g0, g1, ttl::make_shape(n, l), ttl::make_shape(l));
}

TEST(linear_bias_test, test_2)
{
    using R = int32_t;

    using F = nn::ops::add_bias<nn::ops::nhwc>;
    using G0 = nn::experimental::ops::grad::add_bias<nn::ops::nhwc, 0>;
    using G1 = nn::experimental::ops::grad::add_bias<nn::ops::nhwc, 1>;

    G0 g0;
    G1 g1;
    F f;

    const int n = 10;
    const int h = 28;
    const int w = 38;
    const int c = 32;

    test_binary_grad<R>(f, g0, g1, ttl::make_shape(n, h, w, c),
                        ttl::make_shape(c));
}
