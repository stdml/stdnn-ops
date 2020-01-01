#include <ttl/algorithm>
#include <ttl/nn/ops>
#include <ttl/nn/testing>
#include <ttl/tensor>

TEST(softmax_test, test_1)
{
    const auto x = ttl::tensor<float, 1>(2);
    x.data()[0] = 0;
    x.data()[1] = 1;
    const auto y = ttl::tensor<float, 1>(x.shape());
    nn::ops::softmax()(ref(y), view(x));

    const float e = std::exp(1);

    ASSERT_FLOAT_EQ(y.data()[0], 1.0 / (1 + e));
    ASSERT_FLOAT_EQ(y.data()[1], e / (1 + e));
}

template <typename R> struct test_softmax {
    void operator()()
    {
        using softmax = nn::ops::softmax;

        for (int k = 1; k <= 100; ++k) {
            {
                const auto x = ttl::tensor<R, 1>(k);
                const auto y = ttl::tensor<R, 1>(x.shape());
                ttl::fill(ref(x), (R)1.0);
                softmax()(ref(y), view(x));
                for (int i = 0; i < k; ++i) {
                    assert_eq<R>()(y.at(i), 1.0 / k);
                }
            }
            {
                const int n = 2;
                const auto x = ttl::tensor<R, 2>(n, k);
                const auto y = ttl::tensor<R, 2>(x.shape());
                ttl::fill(ref(x), (R)1.0);
                softmax()(ref(y), view(x));
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < k; ++j) {
                        assert_eq<R>()(y.at(i, j), 1.0 / k);
                    }
                }
            }
        }
    }
};

TEST(softmax_test, test_float)
{
    test_softmax<float>()();
    test_softmax<double>()();
}
