#pragma once
#include <algorithm>

#include <ttl/algorithm>
#include <ttl/device>
#include <ttl/nn/bits/kernels/cpu/elementary.hpp>
#include <ttl/nn/bits/kernels/reduce.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class inner_contraction<host_memory, R>
{
  public:
    void operator()(const tensor_ref<R, 1> &y, const tensor_view<R, 2> &x)
    {  // Y[i] = \sum X[i, j]
        std::transform(x.begin(), x.end(), y.begin(),
                       // ttl::sum // FIXME: use ttl::sum directly
                       [](const tensor_view<R, 1> &v) { return ttl::sum(v); });
    }
};

template <typename R>
class outter_contraction<host_memory, R>
{
  public:
    void operator()(const tensor_ref<R, 1> &y, const tensor_view<R, 2> &x)
    {  // Y[j] = \sum X[i, j]
        ttl::fill(y, static_cast<R>(0));
        for (const auto xi : x) {
            kernels::add<host_memory, R>()(y, view(y), xi);
        }
    }
};
}  // namespace ttl::nn::kernels
