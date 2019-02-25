#pragma once
#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/reshape.hpp>

namespace nn::layers
{
template <ttl::rank_t p, ttl::rank_t q> class flatten
{
    using op = ops::flatten<p, q>;

  public:
    template <typename R>
    auto operator()(const ttl::tensor_ref<R, p + q> &x) const
    {
        auto y = nn::ops::new_result<ttl::tensor<R, 2>>(op(), x);
        return nn::layers::make_layer(y);
    }
};
}  // namespace nn::layers
