#pragma once

#include <ttl/nn/bits/layers/call.hpp>
#include <ttl/nn/bits/layers/layer.hpp>
#include <ttl/nn/bits/ops/matmul.hpp>

namespace ttl::nn::layers
{
template <typename Act> class activation
{
  public:
    template <typename R, ttl::rank_t r>
    auto operator()(const ttl::tensor_ref<R, r> &x) const
    {
        auto y = ops::new_result<ttl::tensor<R, r>>(Act(), x);
        return make_layer(y);
    }
};
}  // namespace ttl::nn::layers
