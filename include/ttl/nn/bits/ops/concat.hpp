#pragma once
#include <ttl/nn/bits/kernels/cpu/concat.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename image_order>
class concat_channel4d
{
  public:
    template <typename... S>
    shape<4> operator()(const S &... s) const
    {
        constexpr auto pos = traits::channel_position<image_order>;
        return internal::concat_shape<pos>(s...);
    }

    template <typename R, typename D, typename... T>
    void operator()(const tensor_ref<R, 4, D> &y, const T &... xs) const
    {
        constexpr auto arity = sizeof...(T);
        using kernel = kernels::concat_channel4d<D, image_order, arity, R>;
        return kernel()(y, xs...);
    }
};
}  // namespace ttl::nn::ops
