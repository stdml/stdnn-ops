#pragma once
#include <ttl/nn/bits/kernels/cpu/blas.hpp>
#include <ttl/nn/bits/ops/blas.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <arity_t>
class matmul;

template <>
class matmul<0> : public basic_gradient_function<ops::matmul, 0>
{
    using P = basic_gradient_function<ops::matmul, 0>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void
    operator()(const tensor_ref<R, 2, D> &gx, const tensor_view<R, 2, D> &gz,
               const tensor_view<R, 2, D> &z, const tensor_view<R, 2, D> &x,
               const tensor_view<R, 2, D> &y) const
    {
        using E = typename engines::default_blas<D>::type;
        kernels::mmt<D, E, R>()(gx, gz, y);
    }
};

template <>
class matmul<1> : public basic_gradient_function<ops::matmul, 1>
{
    using P = basic_gradient_function<ops::matmul, 1>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void
    operator()(const tensor_ref<R, 2, D> &gy, const tensor_view<R, 2, D> &gz,
               const tensor_view<R, 2, D> &z, const tensor_view<R, 2, D> &x,
               const tensor_view<R, 2, D> &y) const
    {
        using E = typename engines::default_blas<D>::type;
        kernels::mtm<D, E, R>()(gy, x, gz);
    }
};
}  // namespace ttl::nn::ops::grad
