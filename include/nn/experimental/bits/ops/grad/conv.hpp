#pragma once
#include <nn/bits/ops/col2im.hpp>
#include <nn/bits/ops/conv.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops::grad
{
template <typename image_order, typename filter_order, int> class conv;

template <>
class conv<nn::ops::nhwc, nn::ops::rscd, 0>
    : public nn::ops::conv_trait<nn::ops::hw>
{
    using F = nn::ops::conv<nn::ops::nhwc, nn::ops::rscd>;
    const F f_;

  public:
    conv(const F &f) : conv_trait(f), f_(f) {}

    shape<4> operator()(const shape<4> &gz, const shape<4> &z,
                        const shape<4> &x, const shape<4> &y) const
    {
        return nn::ops::gradient_shape<0>(f_, gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &gx,
                    const ttl::tensor_view<R, 4> &gz,
                    const ttl::tensor_view<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
        // check_shape(*this, gx, gz, z, x, y);

        const auto [r, s] =
            nn::ops::filter_shape<nn::ops::rscd>(y.shape()).dims();
        using upper_op = nn::ops::im2col<nn::ops::hwc, nn::ops::hwrsc>;
        const auto upper = nn::ops::internal::make_batched(
            upper_op(h_trait_.get_sample(r), w_trait_.get_sample(s)));

        ttl::tensor<R, 6> gx_upper(upper(x.shape()));  // FIXME: get from pool

        nn::engines::linag<nn::engines::default_engine>::mmt(
            nn::ops::as_matrix<3, 1>(gz),  //
            nn::ops::as_matrix<3, 1>(y),
            nn::ops::as_matrix<3, 3>(ref(gx_upper)));

        using lower_op = nn::ops::col2im<nn::ops::hwc, nn::ops::hwrsc>;
        const auto lower = nn::ops::internal::make_batched(
            lower_op(h_trait_.get_sample(r), w_trait_.get_sample(s)));
        lower(gx, view(gx_upper));
    }
};

template <>
class conv<nn::ops::nhwc, nn::ops::rscd, 1>
    : public nn::ops::conv_trait<nn::ops::hw>
{
    using F = nn::ops::conv<nn::ops::nhwc, nn::ops::rscd>;
    const F f_;

  public:
    conv(const F &f) : conv_trait(f), f_(f) {}

    shape<4> operator()(const shape<4> &gz, const shape<4> &z,
                        const shape<4> &x, const shape<4> &y) const
    {
        return nn::ops::gradient_shape<1>(f_, gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &gy,
                    const ttl::tensor_view<R, 4> &gz,
                    const ttl::tensor_view<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
        check_shape(*this, gy, gz, z, x, y);

        const auto [r, s] =
            nn::ops::filter_shape<nn::ops::rscd>(y.shape()).dims();
        using upper_op = nn::ops::im2col<nn::ops::hwc, nn::ops::hwrsc>;
        const auto upper = nn::ops::internal::make_batched(
            upper_op(h_trait_.get_sample(r), w_trait_.get_sample(s)));

        ttl::tensor<R, 6> x_upper(upper(x.shape()));  // FIXME: get from pool
        upper(ref(x_upper), view(x));  // FIXME: x_upper may be cached

        nn::engines::linag<nn::engines::default_engine>::mtm(
            nn::ops::as_matrix<3, 3>(view(x_upper)),
            nn::ops::as_matrix<3, 1>(gz),  //
            nn::ops::as_matrix<3, 1>(gy));
    }
};

}  // namespace nn::experimental::ops::grad
