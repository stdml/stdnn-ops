#include <ttl/nn/ops>
#include <ttl/nn/testing>

TEST(summary_test, test_1)
{
    const int n = 10;
    ttl::tensor<float, 1> x(n);
    for (int i = 0; i < n; ++i) { x.data()[i] = i; }

    {
        ttl::nn::ops::scalar_summaries<ttl::nn::ops::summaries::min> op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)0);
    }
    {
        ttl::nn::ops::scalar_summaries<ttl::nn::ops::summaries::max> op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)(n - 1));
    }
    {
        ttl::nn::ops::scalar_summaries<ttl::nn::ops::summaries::sum> op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        const int sum = n * (n - 1) / 2;
        ASSERT_FLOAT_EQ(y.data()[0], (float)(sum));
    }
    {
        ttl::nn::ops::scalar_summaries<ttl::nn::ops::summaries::min,
                                       ttl::nn::ops::summaries::max>
            op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)0);
        ASSERT_FLOAT_EQ(y.data()[1], (float)(n - 1));
    }
    {
        ttl::nn::ops::scalar_summaries<ttl::nn::ops::summaries::min,   //
                                       ttl::nn::ops::summaries::max,   //
                                       ttl::nn::ops::summaries::span,  //
                                       ttl::nn::ops::summaries::mean,  //
                                       ttl::nn::ops::summaries::var,   //
                                       ttl::nn::ops::summaries::std,   //
                                       ttl::nn::ops::summaries::adj_diff_sum>
            op;
        ttl::tensor<float, 1> y(op(x.shape()));
        op(ref(y), view(x));
        ASSERT_FLOAT_EQ(y.data()[0], (float)0);                     // min
        ASSERT_FLOAT_EQ(y.data()[1], (float)(n - 1));               // max
        ASSERT_FLOAT_EQ(y.data()[2], (float)(n - 1));               // span
        ASSERT_FLOAT_EQ(y.data()[3], (float)((n - 1) / 2.0));       // mean
        ASSERT_FLOAT_EQ(y.data()[4], (float)(8.25));                // var
        ASSERT_FLOAT_EQ(y.data()[5], (float)(2.8722813232690143));  // std
        ASSERT_FLOAT_EQ(y.data()[6], (float)(n - 1));  // adj_diff_sum
    }
}
