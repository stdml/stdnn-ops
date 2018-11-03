#pragma once
#include <nn/bits/ops/concat.hpp>

namespace nn::layers
{
template <typename image_order> class concat_channel4d
{
    using op = nn::ops::concat_channel4d<image_order>;

  public:
    template <typename R, typename... T>
    auto operator()(const ttl::tensor_ref<R, 4> &x1, const T &... x) const
    {
        auto y = nn::ops::new_result<ttl::tensor<R, 4>>(op(), x1, x...);
        return nn::layers::make_layer(y);
    }
};
}  // namespace nn::layers
