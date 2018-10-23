#pragma once
#include <experimental/contract>
#include <experimental/new_type>
#include <nn/bits/engines/linag.hpp>
#include <nn/bits/ops/combinators.hpp>
#include <nn/bits/ops/im2col.hpp>
#include <nn/bits/ops/reshape.hpp>
#include <nn/bits/ops/traits.hpp>
#include <stdtensor>

namespace nn::ops
{
template <typename TensorOrder> class conv_trait;
template <typename TensorOrder, typename FilterOrder = rscd> class conv_infer;
template <typename TensorOrder, typename FilterOrder = rscd> class conv;

template <> class conv_trait<hw>
{
  protected:
    struct padding_trait;
    struct stride_trait;
    struct rate_trait;

    using padding_t = std::experimental::new_type<shape<2>, padding_trait>;
    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;
    using rate_t = std::experimental::new_type<shape<2>, rate_trait>;

    static constexpr auto default_padding = padding_t(0, 0);
    static constexpr auto default_stride = stride_t(1, 1);
    static constexpr auto default_rate = rate_t(1, 1);

    const padding_t padding_;
    const stride_t stride_;
    const rate_t rate_;

    static size_t output_size(const size_t input_size, const size_t kernel_size,
                              const size_t pad, const size_t stride,
                              const size_t rate)
    {
        const size_t padded_size = input_size + 2 * pad;
        contract_assert(kernel_size >= 1);
        const size_t patch_size = (kernel_size - 1) * rate + 1;
        contract_assert(padded_size >= patch_size);
        contract_assert((padded_size - patch_size) % stride == 0);
        return (padded_size - patch_size) / stride + 1;
    }

  public:
    static padding_t padding(int r, int s) { return padding_t(r, s); };
    static stride_t stride(int r, int s) { return stride_t(r, s); };

    conv_trait() : conv_trait(default_padding) {}

    conv_trait(const padding_t &padding) : conv_trait(padding, default_stride)
    {
    }

    conv_trait(const stride_t &stride) : conv_trait(default_padding, stride) {}

    conv_trait(const padding_t &padding, const stride_t &stride)
        : conv_trait(padding, stride, default_rate)
    {
    }

    conv_trait(const padding_t &padding, const stride_t &stride,
               const rate_t &rate)
        : padding_(padding), stride_(stride), rate_(rate)
    {
    }
};

template <> class conv_infer<nhwc, rscd> : public conv_trait<hw>
{
    using conv_trait::conv_trait;

  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        const auto [n, h, w, c] = x.dims;
        const auto [r, s, _c, d] = y.dims;
        contract_assert(c == _c);

        const auto [pad_h, pad_w] = padding_.dims;
        const auto [stride_h, stride_w] = stride_.dims;
        const auto [rate_h, rate_w] = rate_.dims;

        const size_t h_ = output_size(h, r, pad_h, stride_h, rate_h);
        const size_t w_ = output_size(w, s, pad_w, stride_w, rate_w);
        return shape<4>(n, h_, w_, d);
    }
};

template <> class conv_infer<nchw, rscd> : public conv_trait<hw>
{
    using conv_trait::conv_trait;

  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        const auto [n, c, h, w] = x.dims;
        const auto [r, s, _c, d] = y.dims;
        contract_assert(c == _c);

        const auto [pad_h, pad_w] = padding_.dims;
        const auto [stride_h, stride_w] = stride_.dims;
        const auto [rate_h, rate_w] = rate_.dims;

        const size_t h_ = output_size(h, r, pad_h, stride_h, rate_h);
        const size_t w_ = output_size(w, s, pad_w, stride_w, rate_w);
        return shape<4>(n, d, h_, w_);
    }
};

template <> class conv<nhwc, rscd> : public conv_infer<nhwc, rscd>
{
    using conv_infer::conv_infer;

  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        return conv_infer::operator()(x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
        const auto [n, h, w, c] = x.shape().dims;
        const auto [r, s, _c, d] = y.shape().dims;
        const auto [_n, h_, w_, _d] = z.shape().dims;
        contract_assert(_n == n);
        contract_assert(_c == c);
        contract_assert(_d == d);
        // [n, h, w, c], [r, s, c, d] -> [n, h', w', d]
        // [n, h', w', r, s, c], [r, s, c, d] -> [n, h', w', d]
        // [n, h, w, c] -> [n, h', w', r, s, c]

        using upper_op = im2col<hwc, hwrsc>;
        const auto upper = internal::make_batched(
            upper_op(upper_op::ksize(r, s),
                     upper_op::padding(padding_.dims[0], padding_.dims[1]),
                     upper_op::stride(stride_.dims[0], stride_.dims[1]),
                     upper_op::rate(rate_.dims[0], rate_.dims[1])));

        ttl::tensor<R, 6> x_upper(upper(x.shape()));
        upper(ref(x_upper), view(x));

        nn::engines::linag<R>::mm(
            as_matrix<3, 3, ttl::tensor_view<R, 2>>(x_upper),
            as_matrix<3, 1, ttl::tensor_view<R, 2>>(y),
            as_matrix<3, 1, ttl::tensor_ref<R, 2>>(z));
    }
};

template <> class conv<nchw, rscd> : public conv_infer<nchw, rscd>
{
    using conv_infer::conv_infer;

  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        return conv_infer::operator()(x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
    }
};

}  // namespace nn::ops
