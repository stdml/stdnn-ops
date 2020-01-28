#pragma once
#include <ttl/nn/bits/kernels/cpu/elementary.hpp>
#include <ttl/nn/bits/ops/elementary.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <arity_t>
class mul;

template <>
class mul<0> : public basic_gradient_function<ops::mul, 0>
{
    using basic_gradient_function::basic_gradient_function;

  public:
    using basic_gradient_function::operator();

    template <typename R, rank_t r, typename D>
    void
    operator()(const tensor_ref<R, r, D> &gx, const tensor_view<R, r, D> &gz,
               const tensor_view<R, r, D> &z, const tensor_view<R, r, D> &x,
               const tensor_view<R, r, D> &y) const
    {
        kernels::mul<D, R>()(gx, gz, y);
    }
};

template <>
class mul<1> : public basic_gradient_function<ops::mul, 1>
{
    using basic_gradient_function::basic_gradient_function;

  public:
    using basic_gradient_function::operator();

    template <typename R, rank_t r, typename D>
    void
    operator()(const tensor_ref<R, r, D> &gy, const tensor_view<R, r, D> &gz,
               const tensor_view<R, r, D> &z, const tensor_view<R, r, D> &x,
               const tensor_view<R, r, D> &y) const
    {
        kernels::mul<D, R>()(gy, gz, x);
    }
};
}  // namespace ttl::nn::ops::grad
