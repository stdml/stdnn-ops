#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/pool.hpp>
#include <ttl/nn/bits/traits/pool_traits.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R, typename algo>
class pool<host_memory, algo, R>
{
  public:
};
}  // namespace ttl::nn::kernels
