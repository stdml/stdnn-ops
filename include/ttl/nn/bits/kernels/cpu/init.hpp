#pragma once
#include <ttl/algorithm>
#include <ttl/device>
#include <ttl/nn/bits/kernels/init.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class zeros<host_memory, R>
{
  public:
    void operator()(const tensor_ref<R, 1> &x) const
    {
        fill(x, static_cast<R>(0));
    }
};

template <typename R>
class ones<host_memory, R>
{
  public:
    void operator()(const tensor_ref<R, 1> &x) const
    {
        fill(x, static_cast<R>(1));
    }
};

template <typename R>
class uniform_constant<host_memory, R>
{
  public:
    void operator()(const tensor_ref<R, 1> &x) const
    {
        static_assert(std::is_floating_point<R>::value);
        const R value = static_cast<R>(1) / static_cast<R>(x.shape().size());
        fill(x, value);
    }
};
}  // namespace ttl::nn::kernels
