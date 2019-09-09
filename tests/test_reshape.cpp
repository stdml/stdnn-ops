#include <ttl/tensor>

#include <nn/bits/ops/reshape.hpp>

#include "testing.hpp"

template <typename R>
void test_flatten(const int n, const int h, const int w, const int c)
{
    ttl::tensor<R, 4> x(n, h, w, c);
    nn::ops::copy_flatten<1, 3> f;
    ttl::tensor<R, 2> y(f(x.shape()));
    ASSERT_EQ(y.shape(), ttl::make_shape(n, h * w * c));
    f(ref(y), view(x));
}

TEST(reshape_test, test1)
{
    test_flatten<int>(10, 28, 28, 3);
    test_flatten<float>(10, 28, 28, 3);
}
