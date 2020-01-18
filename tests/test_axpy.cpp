#include <ttl/algorithm>
#include <ttl/nn/bits/ops/axpy.hpp>
#include <ttl/nn/testing>

TEST(axpy, test1)
{
    using R = int;
    const int n = 10;
    ttl::tensor<R, 0> a;
    ttl::tensor<R, 1> x(10);
    ttl::tensor<R, 1> y(10);

    ttl::fill(ref(a), 2);
    ttl::fill(ref(x), 3);
    ttl::fill(ref(y), 4);

    ttl::nn::ops::axpy op;
    ttl::tensor<R, 1> z(op(a.shape(), x.shape(), y.shape()));
    op(ref(z), view(a), view(x), view(y));

    for (auto i : ttl::range(n)) { ASSERT_EQ(z.data()[i], 10); }
}

namespace ttl
{
template <typename R> ttl::tensor<R, 0> scalar(const R &x)
{
    ttl::tensor<R, 0> t;
    t.data()[0] = x;
    return t;
}
}  // namespace ttl

TEST(axpy, test2)
{
    using R = int;
    const int n = 10;

    ttl::tensor<R, 1> x(10);
    ttl::tensor<R, 1> y(10);
    ttl::tensor<R, 1> z(n);

    ttl::fill(ref(x), 3);
    ttl::fill(ref(y), 4);

    ttl::nn::ops::axpy()(ref(z), view(ttl::scalar(2)), view(x), view(y));

    for (auto i : ttl::range(n)) { ASSERT_EQ(z.data()[i], 10); }
}
