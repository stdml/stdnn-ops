#pragma once

namespace ttl
{
namespace nn
{
namespace kernels
{
template <typename D, typename F, typename R>
class pointwise;

template <typename D, typename R>
class identity;

template <typename D, typename F, typename R>
class binary_pointwise;

template <typename D, typename R>
class add;

template <typename D, typename R>
class sub;

template <typename D, typename R>
class mul;

template <typename D, typename R>
class div;
}  // namespace kernels
}  // namespace nn
}  // namespace ttl
