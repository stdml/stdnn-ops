#include <nn/bits/ops/impl/col2im1d.hpp>
#include <nn/bits/ops/impl/conv1d.hpp>
#include <nn/bits/ops/impl/im2col1d.hpp>
#include <nn/testing>

template <typename R>
void test_col2im1d(const int n, const int ksize = 1, const int stride = 1,
                   const int rate = 1)
{
    nn::traits::linear_padding_trait<int> padding(1);
    nn::ops::im2col1d upper(ksize, stride, rate, padding);
    nn::ops::col2im1d lower(ksize, stride, rate, padding);

    ttl::tensor<R, 1> x(n);
    const auto [m, _k] = upper(x.shape()).dims();
    UNUSED(_k);
    ttl::tensor<R, 2> x_upper(m, ksize);

    const auto x_shape = lower(x_upper.shape());
    ASSERT_EQ(x_shape, x.shape());

    std::iota(x.data(), x.data_end(), 1);
    upper(ref(x_upper), view(x));

    // pprint(view(x));
    // pprint(view(x_upper));

    lower(ref(x), view(x_upper));

    // pprint(view(x));
    // pprint(view(x_upper));
}

TEST(conv1d_test, test_col2im1d)
{
    test_col2im1d<int>(9, 3);
    // test_col2im1d<int>(9, 3);
}

template <typename R>
void test_conv1d(const int n, const int ksize, const int stride = 1,
                 const int rate = 1)
{
    nn::traits::linear_padding_trait<int> padding(1);
    nn::ops::conv1d f(stride, rate, padding);
    nn::ops::im2col1d upper(ksize, stride, rate, padding);

    ttl::tensor<R, 1> x(n);
    ttl::tensor<R, 1> y(ksize);

    ttl::tensor<R, 2> x_upper(upper(x.shape()));
    ttl::tensor<R, 1> z(f(x.shape(), y.shape()));

    std::iota(x.data(), x.data_end(), 1);
    std::iota(y.data(), y.data_end(), 1);
    std::iota(z.data(), z.data_end(), 1);

    f(ref(z), view(x), view(y));
    upper(ref(x_upper), view(x));

    // pprint(view(x));
    // pprint(view(y));
    // pprint(view(z));
    // pprint(view(x_upper));
}

TEST(conv1d_test, test_conv1d)
{
    test_conv1d<int>(9, 3);
    test_conv1d<int>(9, 3, 1, 2);
    test_conv1d<int>(9, 3, 1, 3);
}
