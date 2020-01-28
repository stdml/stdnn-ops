#pragma once
#include <ttl/nn/bits/kernels/cpu/activation.hpp>
#include <ttl/nn/bits/ops/activation.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <arity_t>
class relu;

template <>
class relu<0> : public basic_gradient_function<ops::relu, 0>
{
    using basic_gradient_function::basic_gradient_function;

  public:
    using basic_gradient_function::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &gx,
                    const tensor_view<R, r, D> &gy,
                    const tensor_view<R, r, D> & /* y */,
                    const tensor_view<R, r, D> &x) const
    {
        kernels::relu_grad<D, R>()(flatten(gx), flatten(gy), flatten(x));
    }
};
}  // namespace ttl::nn::ops::grad
