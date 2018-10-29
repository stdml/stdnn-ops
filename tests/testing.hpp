#pragma once
#include <algorithm>
#include <experimental/range>

#include <gtest/gtest.h>

#include <nn/ops>
#include <stdtensor>

using std::experimental::range;

template <typename R, ttl::rank_t r> void fill(const ttl::tensor<R, r> &t, R x)
{
    std::fill(t.data(), t.data() + t.shape().size(), x);
}

template <typename R> struct assert_eq;

template <> struct assert_eq<float> {
    void operator()(float x, float y) { ASSERT_FLOAT_EQ(x, y); }
};

template <> struct assert_eq<double> {
    void operator()(double x, double y) { ASSERT_FLOAT_EQ(x, y); }
};
