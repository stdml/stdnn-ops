#pragma once

#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/matmul.hpp>
#include <nn/bits/ops/traits.hpp>

namespace nn::layers
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
}  // namespace nn::layers
