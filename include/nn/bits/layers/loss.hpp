#pragma once

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
        const auto op = Act();
        auto y = new ttl::tensor<R, 2>(x.shape());

        using T = ttl::tensor<R, 2>;
        using layer_t = layer<T>;
        return layer_t(y);
    }
};
}  // namespace nn::layers
