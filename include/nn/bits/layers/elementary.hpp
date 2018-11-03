#pragma once
#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/elementary.hpp>

namespace nn::layers
{
class identity
{
    using op = ops::identity;

  public:
    template <typename R, ttl::rank_t r>
    auto operator()(const ttl::tensor_ref<R, r> &x) const
    {
        auto y = nn::ops::new_result<ttl::tensor<R, r>>(op(), x);
        return nn::layers::make_layer(y);
    }
};
}  // namespace nn::layers
