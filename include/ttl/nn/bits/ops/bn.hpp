#pragma once
#include <ttl/nn/bits/kernels/cpu/normalization.hpp>
#include <ttl/nn/bits/ops/bias.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
template <typename image_order>
class batch_norm
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &r_mean,
                        const shape<1> &r_var) const
    {
        const auto c = traits::channel_size<image_order>(x);
        contract_assert(c == std::get<0>(r_mean.dims()));
        contract_assert(c == std::get<0>(r_var.dims()));
        return x;
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &y, const tensor_view<R, 4, D> &x,
                    const tensor_view<R, 1, D> &rolling_mean,
                    const tensor_view<R, 1, D> &rolling_var) const
    {
        kernels::batch_norm<D, image_order, R>()(y, x, rolling_mean,
                                                 rolling_var);
    }
};

template <typename image_order>
class batch_norm_with_bias
{
    using bn_op = batch_norm<image_order>;

  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &r_mean,
                        const shape<1> &r_var, const shape<1> &beta,
                        const shape<1> &gamma) const
    {
        const auto c = ops::channel_size<image_order>(x);
        contract_assert(c == std::get<0>(r_mean.dims()));
        contract_assert(c == std::get<0>(r_var.dims()));
        contract_assert(c == std::get<0>(beta.dims()));
        contract_assert(c == std::get<0>(gamma.dims()));
        return x;
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &y, const tensor_view<R, 4, D> &x,
                    const tensor_view<R, 1, D> &rolling_mean,
                    const tensor_view<R, 1, D> &rolling_var,
                    const tensor_view<R, 1, D> &beta,
                    const tensor_view<R, 1, D> &gamma) const
    {
        bn_op()(y, x, rolling_mean, rolling_var);
        mul_bias<image_order>()(y, view(y), gamma);
        add_bias<image_order>()(y, view(y), beta);
    }
};
}  // namespace ttl::nn::ops
