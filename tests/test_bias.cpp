#include <ttl/algorithm>
#include <ttl/nn/ops>
#include <ttl/nn/testing>
#include <ttl/tensor>

template <typename Op, typename R>
void test_apply_bias_nhwc(int n, int h, int w, int c, R a, R b, R value)
{
    const auto x = ttl::tensor<R, 4>(n, h, w, c);
    const auto y = ttl::tensor<R, 1>(c);
    ttl::fill(ref(x), a);
    ttl::fill(ref(y), b);

    const auto add_bias = ttl::nn::ops::apply_bias<ttl::nn::ops::nhwc, Op>();
    const auto z = ttl::tensor<int, 4>(add_bias(x.shape(), y.shape()));
    ASSERT_EQ(z.shape(), x.shape());
    add_bias(ref(z), view(x), view(y));

    for (auto k : ttl::range(n)) {
        for (auto i : ttl::range(h)) {
            for (auto j : ttl::range(w)) {
                for (auto l : ttl::range(c)) { ASSERT_EQ(z.at(k, i, j, l), value); }
            }
        }
    }
}

template <typename Op, typename R>
void test_apply_bias_nchw(int n, int h, int w, int c, R a, R b, R value)
{
    const auto x = ttl::tensor<R, 4>(n, c, h, w);
    const auto y = ttl::tensor<R, 1>(c);
    ttl::fill(ref(x), a);
    ttl::fill(ref(y), b);

    const auto add_bias = ttl::nn::ops::apply_bias<ttl::nn::ops::nchw, Op>();
    const auto z = ttl::tensor<int, 4>(add_bias(x.shape(), y.shape()));
    ASSERT_EQ(z.shape(), x.shape());
    add_bias(ref(z), view(x), view(y));

    for (auto k : ttl::range(n)) {
        for (auto l : ttl::range(c)) {
            for (auto i : ttl::range(h)) {
                for (auto j : ttl::range(w)) { ASSERT_EQ(z.at(k, l, i, j), value); }
            }
        }
    }
}

TEST(bias_test, test1)
{
    const uint32_t n = 10;
    const uint32_t c = 3;
    const uint32_t h = 99;
    const uint32_t w = 101;

    using R = int;
    test_apply_bias_nhwc<std::plus<R>, R>(n, h, w, c, 2, 3, 2 + 3);
    test_apply_bias_nhwc<std::multiplies<R>, R>(n, h, w, c, 2, 3, 2 * 3);
    test_apply_bias_nchw<std::plus<R>, R>(n, h, w, c, 2, 3, 2 + 3);
    test_apply_bias_nchw<std::multiplies<R>, R>(n, h, w, c, 2, 3, 2 * 3);
}
