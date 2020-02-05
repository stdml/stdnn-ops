#pragma once
#include <ttl/nn/bits/ops/concat.hpp>

namespace ttl::nn::layers
{
template <typename image_order>
class concat_channel4d
{
    using op = ops::concat_channel4d<image_order>;

  public:
    template <typename R, typename D, typename... T>
    auto operator()(const tensor_view<R, 4, D> &x0, const T &... xs) const
    {
        auto y = ops::new_result<tensor<R, 4, D>>(op(), x0, xs...);
        return make_layer(y);
    }
};
}  // namespace ttl::nn::layers
