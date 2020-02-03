#pragma once
#include <algorithm>
#include <functional>

#include <ttl/device>
#include <ttl/nn/bits/kernels/elementary.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename F, typename R>
class pointwise<host_memory, F, R>
{
    using D = host_memory;
    static constexpr rank_t r = 1;

  public:
    void operator()(const tensor_ref<R, r, D> &y,  //
                    const tensor_view<R, r, D> x) const
    {
        std::transform(x.data(), x.data_end(), y.data(), F());
    }
};

template <typename F, typename R>
using host_pointwise = pointwise<host_memory, F, R>;

template <typename R>
class identity<host_memory, R>
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

template <typename F, typename R>
class binary_pointwise<host_memory, F, R>
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

template <typename F, typename R>
using host_binary_pointwise = binary_pointwise<host_memory, F, R>;

template <typename R>
class add<host_memory, R> : public host_binary_pointwise<std::plus<R>, R>
{
};

template <typename R>
class sub<host_memory, R> : public host_binary_pointwise<std::minus<R>, R>
{
};

template <typename R>
class mul<host_memory, R> : public host_binary_pointwise<std::multiplies<R>, R>
{
};

template <typename R>
class div<host_memory, R> : public host_binary_pointwise<std::divides<R>, R>
{
};
}  // namespace ttl::nn::kernels
