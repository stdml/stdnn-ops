#pragma once
#include <ttl/nn/bits/layers/call.hpp>
#include <ttl/nn/bits/layers/layer.hpp>
#include <ttl/nn/bits/ops/bias.hpp>
#include <ttl/nn/bits/ops/blas.hpp>

namespace ttl::nn::layers
{
class dense_trait
{
  protected:
    const size_t logits_;

  public:
    dense_trait(size_t logits) : logits_(logits) {}

    shape<2> weight_shape(const shape<2> &x) const
    {
        return shape<2>(std::get<1>(x.dims()), logits_);
    }

    shape<1> bias_shape(const shape<2> &x) const { return shape<1>(logits_); }
};

template <typename Act = ops::noop>
class dense : public dense_trait
{
    using dense_trait::dense_trait;

  public:
    template <typename R, typename D, typename Winit = ops::noop,
              typename Binit = ops::noop>
    auto operator()(const tensor_view<R, 2, D> &x,
                    const Winit &w_init = Winit(),
                    const Binit &b_init = Binit()) const
    {
        using T1 = tensor<R, 1, D>;
        using T2 = tensor<R, 2, D>;
        auto w = ops::new_parameter<T2>(weight_shape(x.shape()), w_init);
        auto y = ops::new_result<T2>(ops::matmul(), x, view(*w));
        auto b = ops::new_parameter<T1>(bias_shape(x.shape()), b_init);
        ops::add_bias<ops::hw>()(ref(*y), view(*y), view(*b));
        Act()(ref(*y), view(*y));
        return make_layer(y, w, b);
    }
};
}  // namespace ttl::nn::layers
