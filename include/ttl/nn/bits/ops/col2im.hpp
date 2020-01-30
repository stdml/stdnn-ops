#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/cpu/col2im.hpp>
#include <ttl/nn/bits/traits/conv_traits.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename image_order, typename col_order>
class col2im;

template <>
class col2im<traits::hwc, traits::hwrsc>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

    template <typename dim_t>
    static auto origin_size(dim_t m,
                            const traits::linear_sample_trait<dim_t> &t)
    {
        return (m - 1) * t.get_stride() + t.get_filter().patch_size() -
               t.get_padding()(0);
    }

  public:
    shape<3> operator()(const shape<5> &x) const
    {
        const auto [h_, w_, r, s, c] = x.dims();
        contract_assert_eq(get_ksize(), shape<2>(r, s));
        const auto &[h_trait, w_trait] = samples();
        const int h = origin_size(h_, h_trait);
        const int w = origin_size(w_, w_trait);
        return make_shape(h, w, c);
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 3, D> &y,
                    const tensor_view<R, 5, D> &x) const
    {
        const traits::im2col_trait<traits::hw> &trait = *this;
        (kernels::col2im_2d<D, traits::hwc, traits::hwrsc, R>(trait))(y, x);
    }
};
}  // namespace ttl::nn::ops
