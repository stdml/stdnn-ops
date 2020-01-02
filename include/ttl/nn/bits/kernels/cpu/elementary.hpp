#pragma once
#include <algorithm>

#include <ttl/device>
#include <ttl/nn/bits/kernels/elementary.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class identity<R, host_memory>
{
    using D = host_memory;
    static constexpr rank_t r = 1;

  public:
    void operator()(const tensor_ref<R, r, D> &y,
                    const tensor_view<R, r, D> &x) const
    {
        std::copy(x.data(), x.data_end(), y.data());
    }
};

template <typename R, typename F>
class _binary_pointwise
{
    using D = host_memory;
    static constexpr rank_t r = 1;

  public:
    void operator()(const tensor_ref<R, r, D> &z,  //
                    const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        std::transform(x.data(), x.data_end(), y.data(), z.data(), F());
    }
};

template <typename R>
class add<R, host_memory> : public _binary_pointwise<R, std::plus<R>>
{
};

template <typename R>
class sub<R, host_memory> : public _binary_pointwise<R, std::minus<R>>
{
};

template <typename R>
class mul<R, host_memory> : public _binary_pointwise<R, std::multiplies<R>>
{
};

template <typename R>
class div<R, host_memory> : public _binary_pointwise<R, std::divides<R>>
{
};
}  // namespace ttl::nn::kernels
