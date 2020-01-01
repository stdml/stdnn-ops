#include <ttl/nn/bits/ops/gradients/add.hpp>
#include <ttl/nn/bits/ops/gradients/matmul.hpp>
#include <ttl/nn/bits/ops/gradients/softmax.hpp>
#include <ttl/nn/bits/ops/gradients/xentropy.hpp>
#include <ttl/nn/ops>
#include <ttl/nn/testing>

template <typename R, typename S>
ttl::tensor<R, S::rank> make_tensor(const S &shape)
{
    return ttl::tensor<R, S::rank>(shape);
}

template <typename F, typename G0, typename S0>
void test_unary_grad_shape(const F &f, const G0 g0, const S0 &shape)
{
    using R = float;
    auto x = make_tensor<R>(shape);
    auto y = make_tensor<R>(f(x.shape()));
    auto gy = make_tensor<R>(y.shape());
    auto gx = make_tensor<R>(g0(gy.shape(), y.shape(), x.shape()));
    ASSERT_EQ(x.shape(), gx.shape());
}

template <typename F, typename G0, typename G1, typename S0, typename S1>
void test_binary_grad_shape(const F &f,                //
                            const G0 g0, const G1 g1,  //
                            const S0 &s0, const S1 &s1)
{
    using R = float;
    auto x = make_tensor<R>(s0);
    auto y = make_tensor<R>(s1);
    auto z = make_tensor<R>(f(x.shape(), y.shape()));
    auto gz = make_tensor<R>(z.shape());
    auto gx = make_tensor<R>(g0(gz.shape(), z.shape(), x.shape(), y.shape()));
    auto gy = make_tensor<R>(g1(gz.shape(), z.shape(), x.shape(), y.shape()));
    ASSERT_EQ(x.shape(), gx.shape());
    ASSERT_EQ(y.shape(), gy.shape());
}

TEST(grad_shape_test, test_softmax)
{
    using F = nn::ops::softmax;
    using G0 = nn::ops::grad::softmax<0>;
    F f;
    G0 g0;
    test_unary_grad_shape(f, g0, nn::shape<1>(10));
}

TEST(grad_shape_test, test_add)
{
    using F = nn::ops::add;
    using G0 = nn::ops::grad::add<0>;
    using G1 = nn::ops::grad::add<1>;
    F f;
    G0 g0;
    G1 g1;
    test_binary_grad_shape(f, g0, g1, nn::shape<1>(10), nn::shape<1>(10));
}

TEST(grad_shape_test, test_matmul)
{
    using F = nn::ops::matmul;
    using G0 = nn::ops::grad::matmul<0>;
    using G1 = nn::ops::grad::matmul<1>;
    F f;
    G0 g0;
    G1 g1;
    const int k = 10;
    const int m = 11;
    const int n = 12;
    test_binary_grad_shape(f, g0, g1, nn::shape<2>(k, m), nn::shape<2>(m, n));
}

// TEST(grad_shape_test, test_xentropy)
// {
//     using F = nn::ops::xentropy;
//     using G0 = nn::ops::grad::xentropy<0>;
//     using G1 = nn::ops::grad::xentropy<1>;
//     F f;
//     G0 g0;
//     G1 g1;
//     test_binary_grad_shape(f, g0, g1, nn::shape<2>(10, 100),
//                            nn::shape<2>(10, 100));
// }
