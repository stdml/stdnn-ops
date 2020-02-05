#pragma once
#include <ttl/nn/bits/layers/call.hpp>
#include <ttl/nn/bits/layers/layer.hpp>
#include <ttl/nn/bits/ops/blas.hpp>

namespace ttl::nn::layers
{
template <typename Act>
class activation
{
  public:
    template <typename R, rank_t r, typename D>
    auto operator()(const tensor_ref<R, r, D> &x) const
    {
        auto y = ops::new_result<tensor<R, r, D>>(Act(), x);
        return make_layer(y);
    }
};
}  // namespace ttl::nn::layers
