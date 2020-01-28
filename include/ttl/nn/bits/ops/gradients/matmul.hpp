#pragma once
#include <ttl/nn/bits/kernels/cpu/blas.hpp>
#include <ttl/nn/bits/ops/blas.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <arity_t, typename E = engines::default_engine>
class matmul;

template <typename E>
class matmul<0, E> : public basic_gradient_function<ops::matmul_<E>, 0>
{
    using P = basic_gradient_function<ops::matmul_<E>, 0>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void operator()(const ttl::tensor_ref<R, 2, D> &gx,
                    const ttl::tensor_view<R, 2, D> &gz,
                    const ttl::tensor_view<R, 2, D> &z,
                    const ttl::tensor_view<R, 2, D> &x,
                    const ttl::tensor_view<R, 2, D> &y) const
    {
        kernels::mmt<D, E, R>()(gx, gz, y);
    }
};

template <typename E>
class matmul<1, E> : public basic_gradient_function<ops::matmul_<E>, 1>
{
    using P = basic_gradient_function<ops::matmul_<E>, 1>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void operator()(const ttl::tensor_ref<R, 2, D> &gy,
                    const ttl::tensor_view<R, 2, D> &gz,
                    const ttl::tensor_view<R, 2, D> &z,
                    const ttl::tensor_view<R, 2, D> &x,
                    const ttl::tensor_view<R, 2, D> &y) const
    {
        kernels::mtm<D, E, R>()(gy, x, gz);
    }
};
}  // namespace ttl::nn::ops::grad
