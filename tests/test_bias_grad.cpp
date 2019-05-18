#include <nn/experimental/bits/ops/grad/bias.hpp>
#include <nn/ops>

#include "testing.hpp"

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

    ttl::tensor<R, 2> x(n, l);
    ttl::tensor<R, 1> y(l);
    ttl::tensor<R, 2> z(n, l);

    ttl::tensor<R, 2> gx(x.shape());
    ttl::tensor<R, 1> gy(y.shape());
    ttl::tensor<R, 2> gz(z.shape());

    f(ref(z), view(x), view(y));

    g0(ref(gx), view(gz), view(z), view(x), view(y));
    g1(ref(gy), view(gz), view(z), view(x), view(y));

    // TODO: assert

    // ttl::tensor<R, 2> x1(x.shape());
    // ttl::tensor<R, 1> y1(y.shape());
    // ttl::tensor<R, 2> z1(z.shape());
}
