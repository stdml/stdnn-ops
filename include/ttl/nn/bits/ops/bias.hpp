#pragma once
#include <ttl/nn/bits/kernels/cpu/bias.hpp>
#include <ttl/nn/bits/ops/elementary.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
template <typename image_order, typename F>
class apply_bias
{
    static constexpr auto r = traits::rank_of<image_order>;
    static constexpr auto p = traits::bias_position<image_order>;

  public:
    shape<r> operator()(const shape<r> &x, const shape<1> &y) const
    {
        contract_assert(std::get<p>(x.dims()) == std::get<0>(y.dims()));
        return x;
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, r, D> &z, const tensor_view<R, r, D> &x,
                    const tensor_view<R, 1, D> &y) const
    {
        kernels::apply_bias<D, image_order, F, R>()(z, x, y);
    }
};

template <typename image_order>
using add_bias = apply_bias<image_order, scalar_add>;

template <typename image_order>
using mul_bias = apply_bias<image_order, scalar_mul>;
}  // namespace ttl::nn::ops
