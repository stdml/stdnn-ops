#pragma once
#include <ttl/algorithm>
#include <ttl/device>
#include <ttl/nn/bits/kernels/init.hpp>

namespace ttl::nn::kernels
{
template <typename R>
class zeros<host_memory, R>
{
  public:
    void operator()(const ttl::tensor_ref<R, 1> &x) const
    {
        ttl::fill(x, static_cast<R>(0));
    }
};

template <typename R>
class ones<host_memory, R>
{
  public:
    void operator()(const ttl::tensor_ref<R, 1> &x) const
    {
        ttl::fill(x, static_cast<R>(1));
    }
};
}  // namespace ttl::nn::kernels
