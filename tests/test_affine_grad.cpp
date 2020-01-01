#include <ttl/nn/bits/ops/gradients/add.hpp>
#include <ttl/nn/ops>
#include <ttl/nn/testing>

TEST(affine_grad_test, test_1)
{
    using R = int32_t;

    using F = ttl::nn::ops::add;
    using G0 = ttl::nn::ops::grad::add<0>;
    using G1 = ttl::nn::ops::grad::add<1>;
    G0 g0;
    G1 g1;
    F f;

    int n = 10;

    ttl::tensor<R, 1> x(n);
    ttl::tensor<R, 1> y(n);
    ttl::tensor<R, 1> z(n);

    ttl::tensor<R, 1> x1(x.shape());
    ttl::tensor<R, 1> y1(y.shape());
    ttl::tensor<R, 1> z1(z.shape());

    ttl::tensor<R, 1> dx(x.shape());
    ttl::tensor<R, 1> dy(y.shape());
    ttl::tensor<R, 1> dz(z.shape());

    int seed = 1;
    for (auto _ : range(3)) {
        UNUSED(_);

        gen_test_tensor(x, seed);
        gen_test_tensor(y, seed);
        gen_test_tensor(dz, seed);
        f(ref(z), view(x), view(y));

        g0(ref(dx), view(dz), view(z), view(x), view(y));
        g1(ref(dy), view(dz), view(z), view(x), view(y));

        ttl::nn::ops::add()(ref(x1), view(x), view(dx));
        ttl::nn::ops::add()(ref(y1), view(y), view(dy));
        ttl::nn::ops::add()(ref(z1), view(z), view(dz));

        {
            ttl::tensor<R, 1> z2(z.shape());
            f(ref(z2), view(x), view(y1));
            assert_tensor_eq(view(z1), view(z2));
        }
        {
            ttl::tensor<R, 1> z2(z.shape());
            f(ref(z2), view(x1), view(y));
            assert_tensor_eq(view(z1), view(z2));
        }
    }
}
