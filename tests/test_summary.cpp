#include "testing.hpp"

#include <nn/ops>

TEST(summary_test, test_1)
{
    const int n = 10;
    ttl::tensor<float, 1> x(n);
    for (int i = 0; i < n; ++i) { x.data()[i] = i; }

    {
        nn::ops::scalar_summaries<nn::ops::summaries::min> op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)0);
    }
    {
        nn::ops::scalar_summaries<nn::ops::summaries::max> op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)(n - 1));
    }
    {
        nn::ops::scalar_summaries<nn::ops::summaries::sum> op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        const int sum = n * (n - 1) / 2;
        ASSERT_FLOAT_EQ(y.data()[0], (float)(sum));
    }
    {
        nn::ops::scalar_summaries<nn::ops::summaries::min,
                                  nn::ops::summaries::max>
            op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)0);
        ASSERT_FLOAT_EQ(y.data()[1], (float)(n - 1));
    }
    {
        nn::ops::scalar_summaries<
            nn::ops::summaries::min, nn::ops::summaries::max,
            nn::ops::summaries::mean, nn::ops::summaries::var,
            nn::ops::summaries::std, nn::ops::summaries::adj_diff_sum>
            op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)0);
        ASSERT_FLOAT_EQ(y.data()[1], (float)(n - 1));
        ASSERT_FLOAT_EQ(y.data()[2], (float)((n - 1) / 2.0));
        ASSERT_FLOAT_EQ(y.data()[3], (float)(8.25));
        ASSERT_FLOAT_EQ(y.data()[4], (float)(2.8722813232690143));
        ASSERT_FLOAT_EQ(y.data()[5], (float)(n - 1));
    }
}
