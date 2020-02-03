#pragma once
#include <ttl/nn/bits/kernels/cpu/blas.hpp>
#include <ttl/nn/bits/ops/col2im.hpp>
#include <ttl/nn/bits/ops/conv2d.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <typename image_order, typename filter_order, arity_t>
class conv;

template <>
class conv<traits::nhwc, traits::rscd, 0>
    : public basic_gradient_function<ops::conv<traits::nhwc, traits::rscd>, 0>
{
    using basic_gradient_function::basic_gradient_function;

  public:
    using basic_gradient_function::operator();

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &gx,
                    const tensor_view<R, 4, D> &gz,
                    const tensor_view<R, 4, D> & /* z */,
                    const tensor_view<R, 4, D> & /* x */,
                    const tensor_view<R, 4, D> &y) const
    {
        using E = typename engines::default_blas<D>::type;
        // check_shape(*this, gx, gz, z, x, y);
        const auto [r, s] = ops::filter_shape<traits::rscd>(y.shape()).dims();
        using upper_op = ops::im2col<traits::hwc, traits::hwrsc>;
        using internal::make_batched;
        const auto upper = make_batched(upper_op(f_.h_trait().get_sample(r),  //
                                                 f_.w_trait().get_sample(s)));
        tensor<R, 6, D> gx_upper(upper(gx.shape()));  // FIXME: get from pool
        kernels::mmt<D, E, R>()(ops::as_matrix<3, 3>(ref(gx_upper)),
                                ops::as_matrix<3, 1>(gz),
                                ops::as_matrix<3, 1>(y));
        using lower_op = ops::col2im<traits::hwc, traits::hwrsc>;
        const auto lower = make_batched(lower_op(f_.h_trait().get_sample(r),  //
                                                 f_.w_trait().get_sample(s)));
        lower(gx, view(gx_upper));
    }
};

template <>
class conv<traits::nhwc, traits::rscd, 1>
    : public basic_gradient_function<ops::conv<traits::nhwc, traits::rscd>, 1>
{
    using basic_gradient_function::basic_gradient_function;

  public:
    using basic_gradient_function::operator();

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &gy,
                    const tensor_view<R, 4, D> &gz,
                    const tensor_view<R, 4, D> & /* z */,
                    const tensor_view<R, 4, D> &x,
                    const tensor_view<R, 4, D> & /* y */) const
    {
        using E = typename engines::default_blas<D>::type;
        // check_shape(*this, gy, gz, z, x, y);
        const auto [r, s] = ops::filter_shape<traits::rscd>(gy.shape()).dims();
        using upper_op = ops::im2col<traits::hwc, traits::hwrsc>;
        using internal::make_batched;
        const auto upper = make_batched(upper_op(f_.h_trait().get_sample(r),  //
                                                 f_.w_trait().get_sample(s)));
        tensor<R, 6, D> x_upper(upper(x.shape()));  // FIXME: get from pool
        upper(ref(x_upper), x);  // FIXME: x_upper may be cached
        kernels::mtm<D, E, R>()(ops::as_matrix<3, 1>(gy),
                                ops::as_matrix<3, 3>(view(x_upper)),
                                ops::as_matrix<3, 1>(gz));
    }
};
}  // namespace ttl::nn::ops::grad
