#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/blas.hpp>
#include <ttl/nn/bits/kernels/conv.hpp>
#include <ttl/nn/bits/kernels/cpu/blas.hpp>
#include <ttl/nn/traits>
#include <ttl/tensor>

// FIXME: move them out of ops
#include <ttl/nn/bits/ops/combinators.hpp>
#include <ttl/nn/bits/ops/conv2d_trait.hpp>
#include <ttl/nn/bits/ops/im2col.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>

namespace ttl::nn::kernels
{
template <typename R>
class conv2d<host_memory, traits::nhwc, traits::rscd, R>
    : public ops::conv_trait<traits::hw>
{
    using D = host_memory;
    using conv_trait::conv_trait;

  public:
    conv2d(const conv_trait &t) : conv_trait(t) {}

    void operator()(const tensor_ref<R, 4, D> &z, const tensor_view<R, 4, D> &x,
                    const tensor_view<R, 4, D> &y) const
    {
        // kernels::conv2d<D, nhwc, rscd, R> (*this)(z, x, y);
        // [n, h, w, c], [r, s, c, d] -> [n, h', w', d]
        // [n, h', w', r, s, c], [r, s, c, d] -> [n, h', w', d]
        // [n, h, w, c] -> [n, h', w', r, s, c]
        const auto [r, s] =
            traits::filter_shape<traits::rscd>(y.shape()).dims();
        using upper_op = ops::im2col<traits::hwc, traits::hwrsc>;
        using ops::internal::make_batched;
        const auto upper = make_batched(upper_op(h_trait_.get_sample(r),  //
                                                 w_trait_.get_sample(s)));
        tensor<R, 6, D> x_upper(upper(x.shape()));
        upper(ref(x_upper), x);
        mm<D, engines::default_engine, R>()(ops::as_matrix<3, 1>(z),
                                            ops::as_matrix<3, 3>(view(x_upper)),
                                            ops::as_matrix<3, 1>(y));
    }
};

template <typename R>
class conv2d<host_memory, traits::nchw, traits::dcrs, R>
    : public ops::conv_trait<traits::hw>
{
    using D = host_memory;
    using conv_trait::conv_trait;

  public:
    conv2d(const conv_trait &t) : conv_trait(t) {}

    void operator()(const tensor_ref<R, 4, D> &z, const tensor_view<R, 4, D> &x,
                    const tensor_view<R, 4, D> &y) const
    {
        using upper_op = ops::im2col<traits::hw, traits::rshw>;
        using ops::internal::make_batched;
        const auto [r, s] =
            traits::filter_shape<traits::dcrs>(y.shape()).dims();
        const auto upper = make_batched(upper_op(h_trait_.get_sample(r),  //
                                                 w_trait_.get_sample(s)));
        tensor<R, 5, D> x_upper(upper(x.shape().template subshape<1>()));
        const auto n = traits::batch_size<traits::nchw>(z.shape());
        for (auto l : range(n)) {
            upper(ref(x_upper), x[l]);
            mm<D, engines::default_engine, R>()(
                ops::as_matrix<1, 2>(z[l]), ops::as_matrix<1, 3>(y),
                ops::as_matrix<3, 2>(view(x_upper)));
        }
    }
};
}  // namespace ttl::nn::kernels
