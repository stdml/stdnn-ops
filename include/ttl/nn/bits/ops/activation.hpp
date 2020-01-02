#pragma once
#include <cmath>

#include <algorithm>

#include <ttl/nn/bits/kernels/activation.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <rank_t r, typename R, typename D>
class softmax_impl;

template <typename R, typename D>
class softmax_impl<1, R, D>
{
    const R eps_;

  public:
    softmax_impl(R eps) : eps_(eps) {}

    void operator()(const tensor_ref<R, 1, D> &y,
                    const tensor_view<R, 1, D> &x) const
    {
        (kernels::softmax<R, D>(eps_))(y, x);
    }
};

template <typename R, typename D>
class softmax_impl<2, R, D>
{
    const R eps_;

  public:
    softmax_impl(R eps) : eps_(eps) {}

    void operator()(const tensor_ref<R, 2, D> &y,
                    const tensor_view<R, 2, D> &x) const
    {
        const auto op = kernels::softmax<R, D>(eps_);
        for (auto i : range<0>(x)) { op(y[i], x[i]); }
    }
};

class softmax : public endofunction
{
  public:
    using endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &y, const tensor_view<R, r, D> &x,
                    R eps = static_cast<R>(1e-6)) const
    {
        (softmax_impl<r, R, D>(eps))(y, x);
    }
};

struct relu_scalar {
    template <typename R>
    R operator()(R x)
    {
        return x > 0 ? x : 0.0;
    }
};

using relu = relu_scalar;
// TODO: leaky relu
}  // namespace ttl::nn::ops
