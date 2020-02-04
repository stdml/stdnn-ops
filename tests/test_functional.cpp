#include <ttl/nn/bits/ops/elementary.hpp>
#include <ttl/nn/bits/ops/utility.hpp>
#include <ttl/nn/experimental/functional>

#include <gtest/gtest.h>

TEST(functional_test, test_invoke)
{
    {
        ttl::tensor<int, 1> x(100);
        using R = float;
        ttl::tensor<R, 2> l =
            ttl::nn::invoke<R>(ttl::nn::ops::onehot(10), ttl::view(x));
        ASSERT_EQ(l.shape(), ttl::make_shape(100, 10));
    }
    {
        ttl::tensor<int, 1> x(100);
        ttl::tensor<int, 1> y(100);
        auto z =
            ttl::nn::invoke(ttl::nn::ops::add(), ttl::view(x), ttl::view(y));
        ASSERT_EQ(z.shape(), ttl::make_shape(100));
    }
}
