#include <ttl/experimental/show>
#include <ttl/nn/bits/ops/impl/col2im1d.hpp>
#include <ttl/nn/bits/ops/impl/conv1d.hpp>
#include <ttl/nn/bits/ops/impl/im2col1d.hpp>
#include <ttl/nn/testing>

template <typename R>
void test_col2im1d(const int n, const int ksize = 1, const int stride = 1,
                   const int rate = 1)
{
    nn::traits::linear_padding_trait<int> padding(1);
    nn::ops::im2col1d upper(ksize, stride, rate, padding);
    nn::ops::col2im1d lower(ksize, stride, rate, padding);

    ttl::tensor<R, 1> x(n);
    ttl::tensor<R, 1> x1(n);
    ttl::tensor<R, 1> c(n);
    const auto [m, _k] = upper(x.shape()).dims();
    UNUSED(_k);
    ttl::tensor<R, 2> x_upper(m, ksize);

    const auto x_shape = lower(x_upper.shape());
    ASSERT_EQ(x_shape, x.shape());

    std::iota(x.data(), x.data_end(), 1);
    upper(ref(x_upper), view(x));   // x -> x_upper
    lower(ref(x1), view(x_upper));  // x_upper -> x1

    std::map<R, int> counts;
    for (auto i : ttl::range(x_upper.shape().size())) {
        ++counts[x_upper.data()[i]];
    }
    for (auto i : ttl::range(x.shape().size())) {
        const R xi = x.data()[i];
        ASSERT_EQ(xi * counts[xi], x1.data()[i]);
    }
    // std::cout << ttl::show(view(x));
    // std::cout << ttl::show(view(x1));
    // std::cout << ttl::show(view(x_upper));
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
    ttl::tensor<R, 1> z1(z.shape());

    std::iota(x.data(), x.data_end(), 1);
    std::iota(y.data(), y.data_end(), 1);
    std::iota(z.data(), z.data_end(), 1);

    f(ref(z), view(x), view(y));
    upper(ref(x_upper), view(x));

    using la = nn::engines::linag<nn::engines::plain>;
    la::mv(view(x_upper), view(y), ref(z1));

    assert_bytes_eq(view(z), view(z1));

    // std::cout << ttl::show(view(x));
    // std::cout << ttl::show(view(y));
    // std::cout << ttl::show(view(z));
    // std::cout << ttl::show(view(x_upper));
    // std::cout << std::endl;
}

TEST(conv1d_test, test_conv1d)
{
    test_conv1d<int>(9, 3);
    test_conv1d<int>(9, 3, 1, 2);
    test_conv1d<int>(9, 3, 1, 3);
}
