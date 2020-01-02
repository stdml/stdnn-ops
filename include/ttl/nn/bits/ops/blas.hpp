#pragma once
#include <ttl/nn/bits/engines/config.hpp>
#include <ttl/nn/bits/kernels/blas.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class axpy
{
  public:
    template <rank_t r>
    shape<r> operator()(const shape<0> &a, const shape<r> &x,
                        const shape<r> &y) const
    {
        contract_assert_eq(x, y);
        return x;
    }

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &z, const tensor_view<R, 0, D> &a,
                    const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        kernels::axpy<R, D, engines::plain>()(flatten(z), a, flatten(x),
                                              flatten(y));
    }
};

template <typename E>
class matmul_
{
  public:
    shape<2> operator()(const shape<2> &x, const shape<2> &y) const
    {
        const auto [n, m] = x.dims();
        const auto [_m, k] = y.dims();
        contract_assert(m == _m);
        return shape<2>(n, k);
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 2, D> &z, const tensor_view<R, 2, D> &x,
                    const tensor_view<R, 2, D> &y) const
    {
        kernels::mm<R, D, E>()(z, x, y);
    }
};

using matmul = matmul_<nn::engines::default_engine>;
}  // namespace ttl::nn::ops
