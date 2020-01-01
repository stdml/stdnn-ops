#pragma once
#include <ttl/nn/bits/layers/call.hpp>
#include <ttl/nn/bits/layers/layer.hpp>
#include <ttl/nn/bits/ops/elementary.hpp>

namespace ttl::nn::layers
{
class identity
{
    using op = ops::identity;

  public:
    template <typename R, ttl::rank_t r>
    auto operator()(const ttl::tensor_ref<R, r> &x) const
    {
        auto y = ttl::nn::ops::new_result<ttl::tensor<R, r>>(op(), x);
        return nn::layers::make_layer(y);
    }
};
}  // namespace ttl::nn::layers
