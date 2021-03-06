#pragma once
#include <ttl/nn/bits/layers/call.hpp>
#include <ttl/nn/bits/layers/layer.hpp>
#include <ttl/nn/bits/ops/pool.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::layers
{
template <typename image_order>
class pool_trait;

template <>
class pool_trait<ops::hw>
{
    using op_trait_t = ops::pool_trait<ops::hw>;

    struct ksize_trait;
    struct stride_trait;

    using ksize_t = std::experimental::new_type<shape<2>, ksize_trait>;
    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;

    using padding_policy =
        traits::linear_sample_trait<uint32_t>::padding_policy;

    const ksize_t ksize_;
    const padding_policy padding_;
    const stride_t stride_;

  public:
    static ksize_t ksize(int r, int s) { return ksize_t(r, s); };
    static stride_t stride(int r, int s) { return stride_t(r, s); };

    static padding_policy padding_same()
    {
        return traits::linear_sample_trait<uint32_t>::padding_same();
    }

    static padding_policy padding_valid()
    {
        return traits::linear_sample_trait<uint32_t>::padding_valid();
    }

    pool_trait() : pool_trait(ksize(2, 2)) {}

    pool_trait(const ksize_t &ksize) : pool_trait(ksize, padding_same()) {}

    pool_trait(const ksize_t &ksize, const padding_policy &padding)
        : pool_trait(ksize, padding, stride(ksize.dims()[0], ksize.dims()[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const padding_policy &padding,
               const stride_t &stride)
        : ksize_(ksize), padding_(padding), stride_(stride)
    {
    }

    op_trait_t op_trait(const shape<2> &x) const
    {
        return op_trait_t(
            op_trait_t::ksize(ksize_.dims()[0], ksize_.dims()[1]),
            op_trait_t::padding(
                padding_(ksize_.dims()[0], stride_.dims()[0], 1, x.dims()[0]),
                padding_(ksize_.dims()[1], stride_.dims()[1], 1, x.dims()[1])),
            op_trait_t::stride(stride_.dims()[0], stride_.dims()[1]));
    }
};

template <typename pool_algo, typename image_order = traits::nhwc>
class pool : public pool_trait<traits::hw>
{
    using pool_trait::pool_trait;
    using pool_op = ops::pool<pool_algo, image_order>;

  public:
    template <typename R, typename D>
    auto operator()(const tensor_view<R, 4, D> &x) const
    {
        auto y = ops::new_result<tensor<R, 4, D>>(
            pool_op(op_trait(ops::image_shape<image_order>(x.shape()))), x);
        return make_layer(y);
    }
};
}  // namespace ttl::nn::layers
