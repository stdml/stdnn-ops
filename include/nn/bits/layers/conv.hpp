#pragma once
#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/bias.hpp>
#include <nn/bits/ops/conv.hpp>
#include <nn/bits/ops/noop.hpp>
#include <nn/bits/ops/traits.hpp>

namespace nn::layers
{
class conv_layer_trait : public ops::conv_trait<ops::hw>
{
    using conv_trait::conv_trait;

  protected:
    struct ksize_trait;
    using ksize_t = std::experimental::new_type<shape<2>, ksize_trait>;

    const ksize_t ksize_;
    const size_t n_filters_;

  public:
    static ksize_t ksize(int r, int s) { return ksize_t(r, s); };

    conv_layer_trait(const ksize_t &ksize, size_t n_filters)
        : ksize_(ksize), n_filters_(n_filters)
    {
    }

    conv_layer_trait(const ksize_t &ksize, size_t n_filters,
                     const padding_t &padding)
        : conv_trait(padding), ksize_(ksize), n_filters_(n_filters)
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
};

template <typename image_order = ops::nhwc, typename filter_order = ops::rscd,
          bool with_bias = true, typename Act = nn::ops::noop>
class conv;

template <typename image_order, typename filter_order, typename Act>
class conv<image_order, filter_order, false, Act> : public conv_layer_trait
{
    using conv_layer_trait::conv_layer_trait;
    using conv_op = nn::ops::conv<image_order, filter_order>;

  public:
    template <typename R, typename Winit = ops::noop>
    auto operator()(const ttl::tensor_ref<R, 4> &x,
                    const Winit &w_init = Winit()) const
    {
        auto w = ops::new_parameter<ttl::tensor<R, 4>>(
            filter_shape<image_order, filter_order>(x.shape()), w_init);
        auto y = ops::new_result<ttl::tensor<R, 4>>(
            conv_op(padding_, stride_, rate_), x, *w);

        Act()(ref(*y), view(*y));
        return make_layer(y, w);
    }
};

template <typename image_order, typename filter_order, typename Act>
class conv<image_order, filter_order, true, Act> : public conv_layer_trait
{
    using conv_layer_trait::conv_layer_trait;
    using conv_op = nn::ops::conv<image_order, filter_order>;

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
            conv_op(padding_, stride_, rate_), x, *w);

        using add_bias = nn::ops::apply_bias<image_order, std::plus<R>>;
        auto b = ops::new_parameter<ttl::tensor<R, 1>>(bias_shape(x.shape()),
                                                       b_init);
        add_bias()(ref(*y), view(*y), view(*b));
        Act()(ref(*y), view(*y));
        return make_layer(y, w, b);
    }
};

}  // namespace nn::layers
