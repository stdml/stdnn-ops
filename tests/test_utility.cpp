#include <ttl/nn/bits/ops/utility.hpp>
#include <ttl/nn/testing>

// TODO: test other utility ops

TEST(utility_test, test_argmax)
{
    ttl::nn::ops::argmax f;
    {
        ttl::tensor<float, 1> x(10);
        ttl::tensor<int, 0> y;
        f(ttl::ref(y), ttl::view(x));
    }
    {
        ttl::tensor<float, 2> x(10, 100);
        ttl::tensor<int, 1> y(f(x.shape()));
        ASSERT_EQ(y.shape(), ttl::make_shape(10));
        f(ttl::ref(y), ttl::view(x));
    }
}

TEST(utility_test, test_similarity)
{
    ttl::nn::ops::similarity op;
    const int n = 100;
    ttl::tensor<int, 1> x(n);
    ttl::tensor<int, 1> y(n);

    std::iota(x.data(), x.data_end(), 1);
    std::iota(y.data(), y.data_end(), 1);
    y.data()[0] += 1;

    ttl::tensor<float, 0> z;
    op(ref(z), view(x), view(y));

    ASSERT_FLOAT_EQ(z.data()[0], 0.99);
}
