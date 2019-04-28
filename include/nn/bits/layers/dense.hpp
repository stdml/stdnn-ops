#pragma once

#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/bias.hpp>
#include <nn/bits/ops/matmul.hpp>
#include <nn/bits/ops/traits.hpp>

namespace nn::layers
{
class dense_trait
{
  protected:
    const size_t logits_;

    using matmul = nn::ops::matmul;

  public:
    dense_trait(size_t logits) : logits_(logits) {}

    shape<2> weight_shape(const shape<2> &x) const
    {
        return shape<2>(x.dims[1], logits_);
    }

    shape<1> bias_shape(const shape<2> &x) const { return shape<1>(logits_); }
};

template <typename Act = ops::noop> class dense : public dense_trait
{
    using dense_trait::dense_trait;

  public:
    template <typename R, typename Winit = ops::noop,
              typename Binit = ops::noop>
    auto operator()(const ttl::tensor_ref<R, 2> &x,
                    const Winit &w_init = Winit(),
                    const Binit &b_init = Binit()) const
    {
        auto w = ops::new_parameter<ttl::tensor<R, 2>>(weight_shape(x.shape()),
                                                       w_init);
        auto y = ops::new_result<ttl::tensor<R, 2>>(matmul(), x, *w);

        auto b = ops::new_parameter<ttl::tensor<R, 1>>(bias_shape(x.shape()),
                                                       b_init);
        ops::add_bias<ops::hw>()(ref(*y), view(*y), view(*b));
        Act()(ref(*y), view(*y));
        return make_layer(y, w, b);
    }
};

}  // namespace nn::layers
