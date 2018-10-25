#pragma once
#include <nn/common.hpp>

namespace nn::ops
{
struct nhwc;
struct nchw;

struct chw;
struct hwc;
struct hw;

struct rscd;  // for conv kernel order

// for im2col output
struct hwrs;
struct hwrsc;

struct rshw;

template <typename T> ttl::rank_t batch_position;
template <> constexpr ttl::rank_t batch_position<nchw> = 0;
template <> constexpr ttl::rank_t batch_position<nhwc> = 0;

template <typename T> ttl::rank_t channel_position;
template <> constexpr ttl::rank_t channel_position<chw> = 0;
template <> constexpr ttl::rank_t channel_position<hwc> = 2;
template <> constexpr ttl::rank_t channel_position<nchw> = 1;
template <> constexpr ttl::rank_t channel_position<nhwc> = 3;

template <typename T> ttl::rank_t height_position;
template <> constexpr ttl::rank_t height_position<chw> = 1;
template <> constexpr ttl::rank_t height_position<hwc> = 0;
template <> constexpr ttl::rank_t height_position<nchw> = 2;
template <> constexpr ttl::rank_t height_position<nhwc> = 1;

template <typename T> ttl::rank_t width_position;
template <> constexpr ttl::rank_t width_position<chw> = 2;
template <> constexpr ttl::rank_t width_position<hwc> = 1;
template <> constexpr ttl::rank_t width_position<nchw> = 3;
template <> constexpr ttl::rank_t width_position<nhwc> = 2;

template <typename order, ttl::rank_t r>
shape<4>::dimension_type batch_size(const shape<r> &x)
{
    return std::get<batch_position<order>>(x.dims);
}

template <typename order, ttl::rank_t r>
shape<4>::dimension_type channel_size(const shape<r> &x)
{
    return std::get<channel_position<order>>(x.dims);
}

template <typename order, ttl::rank_t r> shape<2> image_shape(const shape<r> &x)
{
    return shape<2>(std::get<height_position<order>>(x.dims),
                    std::get<width_position<order>>(x.dims));
}

template <typename T> ttl::rank_t filter_height_position;
template <> constexpr ttl::rank_t filter_height_position<rscd> = 0;

template <typename T> ttl::rank_t filter_width_position;
template <> constexpr ttl::rank_t filter_width_position<rscd> = 1;

template <typename order, ttl::rank_t r>
shape<2> filter_shape(const shape<r> &s)
{
    return shape<2>(std::get<filter_height_position<order>>(s.dims),
                    std::get<filter_width_position<order>>(s.dims));
}

namespace internal
{
template <typename order> struct batched_image_shape_impl;

template <> struct batched_image_shape_impl<nhwc> {
    using dim_t = shape<4>::dimension_type;
    shape<4> operator()(dim_t n, const shape<2> &shp, dim_t c) const
    {
        const auto [h, w] = shp.dims;
        return shape<4>(n, h, w, c);
    }
};

template <> struct batched_image_shape_impl<nchw> {
    using dim_t = shape<4>::dimension_type;
    shape<4> operator()(dim_t n, const shape<2> &shp, dim_t c) const
    {
        const auto [h, w] = shp.dims;
        return shape<4>(n, c, h, w);
    }
};

template <typename order> struct conv_filter_shape_impl;

template <> struct conv_filter_shape_impl<rscd> {
    using dim_t = shape<4>::dimension_type;
    shape<4> operator()(dim_t c, const shape<2> &shp, dim_t d) const
    {
        const auto [r, s] = shp.dims;
        return shape<4>(r, s, c, d);
    }
};

}  // namespace internal

template <typename order>
shape<4> batched_image_shape(shape<4>::dimension_type n, const shape<2> &shp,
                             shape<4>::dimension_type c)
{
    return internal::batched_image_shape_impl<order>()(n, shp, c);
}

template <typename order>
shape<4> conv_filter_shape(shape<4>::dimension_type c, const shape<2> &shp,
                           shape<4>::dimension_type d)
{
    return internal::batched_image_shape_impl<order>()(c, shp, d);
}
}  // namespace nn::ops
