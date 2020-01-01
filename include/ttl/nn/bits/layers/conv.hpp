#pragma once
#include <ttl/nn/bits/layers/call.hpp>
#include <ttl/nn/bits/layers/layer.hpp>
#include <ttl/nn/bits/ops/bias.hpp>
#include <ttl/nn/bits/ops/conv2d.hpp>
#include <ttl/nn/bits/ops/noop.hpp>
#include <ttl/nn/bits/traits/multi_linear_sample.hpp>
#include <ttl/nn/traits>

namespace nn::layers
{
class conv_trait
{
    using op_trait_t = ops::conv_trait<ops::hw>;

    struct ksize_trait;
    struct stride_trait;
    struct rate_trait;

    using ksize_t = std::experimental::new_type<shape<2>, ksize_trait>;
    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;
    using rate_t = std::experimental::new_type<shape<2>, rate_trait>;

    using padding_policy = traits::linear_sample_trait<size_t>::padding_policy;

    const size_t n_filters_;
    const ksize_t ksize_;
    const padding_policy padding_;
    const stride_t stride_;
    const rate_t rate_;

  public:
    static ksize_t ksize(int r, int s) { return ksize_t(r, s); };
    static stride_t stride(int r, int s) { return stride_t(r, s); };
    static rate_t rate(int r, int s) { return rate_t(r, s); };

    static padding_policy padding_same()
    {
        return traits::linear_sample_trait<size_t>::padding_same();
    }

    static padding_policy padding_valid()
    {
        return traits::linear_sample_trait<size_t>::padding_valid();
    }

    conv_trait(size_t n_filters, const ksize_t &ksize)
        : conv_trait(n_filters, ksize,
                     padding_valid())  // FIXME: maybe use same as default
    {
    }

    conv_trait(size_t n_filters, const ksize_t &ksize,
               const padding_policy &padding)
        : conv_trait(n_filters, ksize, padding, stride(1, 1))
    {
    }

    conv_trait(size_t n_filters, const ksize_t &ksize,
               const padding_policy &padding, const stride_t &stride)
        : conv_trait(n_filters, ksize, padding, stride, rate(1, 1))
    {
    }

    conv_trait(size_t n_filters, const ksize_t &ksize,
               const padding_policy &padding, const stride_t &stride,
               const rate_t &rate)
        : n_filters_(n_filters),
          ksize_(ksize),
          padding_(padding),
          stride_(stride),
          rate_(rate)
    {
    }

    template <typename image_order, typename filter_order>
    shape<4> filter_shape(const shape<4> &x) const
    {
        const auto c = ops::channel_size<image_order>(x);
        return ops::conv_filter_shape<filter_order>(c, ksize_, n_filters_);
    }

    shape<1> bias_shape(const shape<4> &x) const
    {
        return shape<1>(n_filters_);
    }

    op_trait_t op_trait(const shape<2> &x) const
    {
        return op_trait_t(
            op_trait_t::padding(padding_(ksize_.dims()[0], stride_.dims()[0],
                                         rate_.dims()[0], x.dims()[0]),
                                padding_(ksize_.dims()[1], stride_.dims()[1],
                                         rate_.dims()[1], x.dims()[1])),
            op_trait_t::stride(stride_.dims()[0], stride_.dims()[1]),
            op_trait_t::rate(rate_.dims()[0], rate_.dims()[1]));
    }
};

template <typename image_order = ops::nhwc, typename filter_order = ops::rscd,
          bool with_bias = true, typename Act = ops::noop>
class conv;

template <typename image_order, typename filter_order, typename Act>
class conv<image_order, filter_order, false, Act> : public conv_trait
{
    using conv_trait::conv_trait;
    using conv_op = ops::conv<image_order, filter_order>;

  public:
    template <typename R, typename Winit = ops::noop>
    auto operator()(const ttl::tensor_ref<R, 4> &x,
                    const Winit &w_init = Winit()) const
    {
        auto w = ops::new_parameter<ttl::tensor<R, 4>>(
            filter_shape<image_order, filter_order>(x.shape()), w_init);
        auto y = ops::new_result<ttl::tensor<R, 4>>(
            conv_op(op_trait(ops::image_shape<image_order>(x.shape()))), x, *w);

        Act()(ref(*y), view(*y));
        return make_layer(y, w);
    }
};

template <typename image_order, typename filter_order, typename Act>
class conv<image_order, filter_order, true, Act> : public conv_trait
{
    using conv_trait::conv_trait;
    using conv_op = ops::conv<image_order, filter_order>;

  public:
    template <typename R, typename Winit = ops::noop,
              typename Binit = ops::noop>
    auto operator()(const ttl::tensor_ref<R, 4> &x,
                    const Winit &w_init = Winit(),
                    const Binit &b_init = Binit()) const
    {
        auto w = ops::new_parameter<ttl::tensor<R, 4>>(
            filter_shape<image_order, filter_order>(x.shape()), w_init);
        auto y = ops::new_result<ttl::tensor<R, 4>>(
            conv_op(op_trait(ops::image_shape<image_order>(x.shape()))), x, *w);
        auto b = ops::new_parameter<ttl::tensor<R, 1>>(bias_shape(x.shape()),
                                                       b_init);
        ops::add_bias<image_order>()(ref(*y), view(*y), view(*b));
        Act()(ref(*y), view(*y));
        return make_layer(y, w, b);
    }
};

}  // namespace nn::layers
