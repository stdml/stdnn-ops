#pragma once
#include <ttl/nn/bits/kernels/conv.hpp>
#include <ttl/nn/bits/kernels/cpu/conv2d.hpp>
#include <ttl/nn/bits/ops/conv2d_trait.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
template <typename image_order = traits::nhwc,
          typename filter_order = traits::rscd>
class conv : public conv_trait<traits::hw>
{
    using conv_trait::conv_trait;

  public:
    conv(const conv_trait &t) : conv_trait(t) {}

    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        return conv_trait::infer<image_order, filter_order>(x, y);
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &z, const tensor_view<R, 4, D> &x,
                    const tensor_view<R, 4, D> &y) const
    {
        (kernels::conv2d<D, image_order, filter_order, R>(*this))(z, x, y);
    }
};
}  // namespace ttl::nn::ops
