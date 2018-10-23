#include <gtest/gtest.h>

#include <nn/ops>
#include <stdtensor>

#include "testing.hpp"

template <typename R> struct test_softmax {
    void operator()()
    {
        using softmax = nn::ops::softmax;

        for (int k = 1; k <= 100; ++k) {
            {
                const auto x = ttl::tensor<R, 1>(k);
                const auto y = ttl::tensor<R, 1>(x.shape());
                fill(x, (R)1.0);
                softmax()(ref(y), view(x));
                for (int i = 0; i < k; ++i) {
                    assert_eq<R>()(y.at(i), 1.0 / k);
                }
            }
            {
                const int n = 2;
                const auto x = ttl::tensor<R, 2>(n, k);
                const auto y = ttl::tensor<R, 2>(x.shape());
                fill(x, (R)1.0);
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
