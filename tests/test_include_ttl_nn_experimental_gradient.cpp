#include <ttl/nn/experimental/gradients>

#include <gtest/gtest.h>

TEST(include_test, test1)
{
    //
}

template <typename F, typename G, typename T>
void test_differentiable_endofunction(const F &f, const G &g, const T &x)
{
    T y(x.shape());
    T gy(y.shape());
    T gx(x.shape());

    f(ttl::ref(y), ttl::view(x));
    g(ttl::ref(gx), ttl::view(gy), ttl::view(y), ttl::view(x));
}

TEST(include_test, test_relu)
{
    using F = ttl::nn::ops::relu;
    using G = ttl::nn::ops::grad::relu<0>;

    F f;
    G g(f);
    ttl::tensor<float, 1> x(10);
    test_differentiable_endofunction(f, g, x);
}

TEST(include_test, test_softmax)
{
    // FIXME: !
    // using F = ttl::nn::ops::softmax;
    // using G = ttl::nn::ops::grad::softmax<0>;

    // F f;
    // G g(f);
    // ttl::tensor<float, 1> x(10);
    // test_differentiable_endofunction(f, g, x);
}
