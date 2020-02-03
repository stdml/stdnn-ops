#pragma once

namespace ttl
{
namespace nn
{
namespace kernels
{
// template <typename D, typename R>
// class conv1d;

template <typename D, typename image_order, typename col_order, typename R>
class col2im_2d;

template <typename D, typename image_order, typename col_order, typename R>
class im2col_2d;

template <typename D, typename image_order, typename filter_order, typename R>
class conv2d;
}  // namespace kernels
}  // namespace nn
}  // namespace ttl
