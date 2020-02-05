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
    template <typename R, rank_t r, typename D>
    auto operator()(const tensor_ref<R, r, D> &x) const
    {
        auto y = ops::new_result<tensor<R, r, D>>(op(), x);
        return make_layer(y);
    }
};
}  // namespace ttl::nn::layers
