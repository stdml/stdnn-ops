#pragma once
#include <ttl/nn/bits/traits/basic_traits.hpp>
#include <ttl/shape>

namespace ttl::nn::traits
{
template <typename T>
ttl::rank_t rank_of;
template <>
inline constexpr ttl::rank_t rank_of<hw> = 2;
template <>
inline constexpr ttl::rank_t rank_of<nchw> = 4;
template <>
inline constexpr ttl::rank_t rank_of<nhwc> = 4;

template <typename T>
ttl::rank_t batch_position;
template <>
inline constexpr ttl::rank_t batch_position<nchw> = 0;
template <>
inline constexpr ttl::rank_t batch_position<nhwc> = 0;

template <typename T>
ttl::rank_t channel_position;
template <>
inline constexpr ttl::rank_t channel_position<chw> = 0;
template <>
inline constexpr ttl::rank_t channel_position<hwc> = 2;
template <>
inline constexpr ttl::rank_t channel_position<nchw> = 1;
template <>
inline constexpr ttl::rank_t channel_position<nhwc> = 3;

template <typename T>
ttl::rank_t height_position;
template <>
inline constexpr ttl::rank_t height_position<chw> = 1;
template <>
inline constexpr ttl::rank_t height_position<hwc> = 0;
template <>
inline constexpr ttl::rank_t height_position<nchw> = 2;
template <>
inline constexpr ttl::rank_t height_position<nhwc> = 1;

template <typename T>
ttl::rank_t width_position;
template <>
inline constexpr ttl::rank_t width_position<chw> = 2;
template <>
inline constexpr ttl::rank_t width_position<hwc> = 1;
template <>
inline constexpr ttl::rank_t width_position<nchw> = 3;
template <>
inline constexpr ttl::rank_t width_position<nhwc> = 2;

template <typename T>
inline constexpr ttl::rank_t bias_position = channel_position<T>;
template <>
inline constexpr ttl::rank_t bias_position<hw> = 1;

template <typename order, ttl::rank_t r>
ttl::shape<4>::dimension_type batch_size(const ttl::shape<r> &x)
{
    return std::get<batch_position<order>>(x.dims());
}

template <typename order, ttl::rank_t r>
ttl::shape<4>::dimension_type channel_size(const ttl::shape<r> &x)
{
    return std::get<channel_position<order>>(x.dims());
}

template <typename order, ttl::rank_t r>
ttl::shape<2> image_shape(const ttl::shape<r> &x)
{
    return ttl::shape<2>(std::get<height_position<order>>(x.dims()),
                         std::get<width_position<order>>(x.dims()));
}

template <typename T>
ttl::rank_t filter_height_position;
template <>
inline constexpr ttl::rank_t filter_height_position<rscd> = 0;
template <>
inline constexpr ttl::rank_t filter_height_position<dcrs> = 2;

template <typename T>
ttl::rank_t filter_width_position;
template <>
inline constexpr ttl::rank_t filter_width_position<rscd> = 1;
template <>
inline constexpr ttl::rank_t filter_width_position<dcrs> = 3;

template <typename T>
ttl::rank_t filter_in_channel_position;
template <>
inline constexpr ttl::rank_t filter_in_channel_position<rscd> = 2;
template <>
inline constexpr ttl::rank_t filter_in_channel_position<dcrs> = 1;

template <typename T>
ttl::rank_t filter_out_channel_position;
template <>
inline constexpr ttl::rank_t filter_out_channel_position<rscd> = 3;
template <>
inline constexpr ttl::rank_t filter_out_channel_position<dcrs> = 0;

template <typename order, ttl::rank_t r>
ttl::shape<2> filter_shape(const ttl::shape<r> &s)
{
    return ttl::shape<2>(std::get<filter_height_position<order>>(s.dims()),
                         std::get<filter_width_position<order>>(s.dims()));
}

template <typename order, ttl::rank_t r>
ttl::shape<4>::dimension_type filter_in_channel_size(const ttl::shape<r> &s)
{
    return std::get<filter_in_channel_position<order>>(s.dims());
}

template <typename order, ttl::rank_t r>
ttl::shape<4>::dimension_type filter_out_channel_size(const ttl::shape<r> &s)
{
    return std::get<filter_out_channel_position<order>>(s.dims());
}

namespace internal
{
template <typename order>
struct batched_image_shape_impl;

template <>
struct batched_image_shape_impl<nhwc> {
    using dim_t = ttl::shape<4>::dimension_type;
    ttl::shape<4> operator()(dim_t n, const ttl::shape<2> &shp, dim_t c) const
    {
        const auto [h, w] = shp.dims();
        return ttl::shape<4>(n, h, w, c);
    }
};

template <>
struct batched_image_shape_impl<nchw> {
    using dim_t = ttl::shape<4>::dimension_type;
    ttl::shape<4> operator()(dim_t n, const ttl::shape<2> &shp, dim_t c) const
    {
        const auto [h, w] = shp.dims();
        return ttl::shape<4>(n, c, h, w);
    }
};

template <typename order>
struct conv_filter_shape_impl;

template <>
struct conv_filter_shape_impl<rscd> {
    using dim_t = ttl::shape<4>::dimension_type;
    ttl::shape<4> operator()(dim_t c, const ttl::shape<2> &shp, dim_t d) const
    {
        const auto [r, s] = shp.dims();
        return ttl::shape<4>(r, s, c, d);
    }
};

template <>
struct conv_filter_shape_impl<dcrs> {
    using dim_t = ttl::shape<4>::dimension_type;
    ttl::shape<4> operator()(dim_t c, const ttl::shape<2> &shp, dim_t d) const
    {
        const auto [r, s] = shp.dims();
        return ttl::shape<4>(d, c, r, s);
    }
};

}  // namespace internal

template <typename order>
ttl::shape<4> batched_image_shape(ttl::shape<4>::dimension_type n,
                                  const ttl::shape<2> &shp,
                                  ttl::shape<4>::dimension_type c)
{
    return internal::batched_image_shape_impl<order>()(n, shp, c);
}

template <typename order>
ttl::shape<4> conv_filter_shape(ttl::shape<4>::dimension_type c,
                                const ttl::shape<2> &shp,
                                ttl::shape<4>::dimension_type d)
{
    return internal::conv_filter_shape_impl<order>()(c, shp, d);
}
}  // namespace ttl::nn::traits
