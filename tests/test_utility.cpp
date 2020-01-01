#include <ttl/nn/bits/ops/utility.hpp>
#include <ttl/nn/testing>

// TODO: test other utility ops

TEST(utility_test, test_similarity)
{
    nn::ops::similarity op;
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
