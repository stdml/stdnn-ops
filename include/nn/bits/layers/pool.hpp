#pragma once
#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/pool.hpp>
#include <nn/bits/ops/traits.hpp>

namespace nn::layers
{

template <typename pool_algo, typename image_order = nn::ops::nhwc>
class pool : public ops::pool_trait<ops::hw>
{
    using pool_trait::pool_trait;
    using pool_op = nn::ops::pool<pool_algo, image_order>;

  public:
    template <typename R> auto operator()(const ttl::tensor_ref<R, 4> &x) const
    {
        auto y = ops::new_result<ttl::tensor<R, 4>>(
            pool_op(get_ksize(), get_padding(), get_stride()), x);
        return make_layer(y);
    }
};
}  // namespace nn::layers
