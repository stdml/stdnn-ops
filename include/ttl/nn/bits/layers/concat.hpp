#pragma once
#include <ttl/nn/bits/ops/concat.hpp>

namespace ttl::nn::layers
{
template <typename image_order> class concat_channel4d
{
    using op = ttl::nn::ops::concat_channel4d<image_order>;

  public:
    template <typename R, typename... T>
    auto operator()(const ttl::tensor_ref<R, 4> &x1, const T &... x) const
    {
        auto y = ttl::nn::ops::new_result<ttl::tensor<R, 4>>(op(), x1, x...);
        return nn::layers::make_layer(y);
    }
};
}  // namespace ttl::nn::layers
