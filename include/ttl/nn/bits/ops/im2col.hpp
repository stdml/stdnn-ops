#pragma once
#include <ttl/nn/bits/kernels/cpu/im2col.hpp>
#include <ttl/nn/bits/traits/conv_traits.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename image_order, typename col_order>
class im2col;

template <>
class im2col<traits::hw, traits::hwrs> : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    shape<4> operator()(const shape<2> &x) const
    {
        return ttl::internal::join_shape(im2col_trait::operator()(x),
                                         get_ksize());
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &y,
                    const tensor_view<R, 2, D> &x) const
    {
        const traits::im2col_trait<traits::hw> &trait = *this;
        (kernels::im2col_2d<D, traits::hw, traits::hwrs, R>(trait))(y, x);
    }
};

template <>
class im2col<traits::hw, traits::rshw> : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    shape<4> operator()(const shape<2> &x) const
    {
        return ttl::internal::join_shape(get_ksize(),
                                         im2col_trait::operator()(x));
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 4, D> &y,
                    const tensor_view<R, 2, D> &x) const
    {
        const traits::im2col_trait<traits::hw> &trait = *this;
        (kernels::im2col_2d<D, traits::hw, traits::rshw, R>(trait))(y, x);
    }
};

// TODO: use vectorize
template <>
class im2col<traits::hwc, traits::hwrsc>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    shape<5> operator()(const shape<3> &x) const
    {
        const auto [r, s] = get_ksize().dims();
        const auto [h, w, c] = x.dims();
        const auto [h_, w_] = im2col_trait::operator()(shape<2>(h, w)).dims();
        return shape<5>(h_, w_, r, s, c);
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 5, D> &y,
                    const tensor_view<R, 3, D> &x) const
    {
        const traits::im2col_trait<traits::hw> &trait = *this;
        (kernels::im2col_2d<D, traits::hwc, traits::hwrsc, R>(trait))(y, x);
    }
};
}  // namespace ttl::nn::ops
