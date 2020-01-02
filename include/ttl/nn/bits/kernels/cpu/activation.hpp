#pragma once
#include <cmath>

#include <algorithm>

#include <ttl/device>
#include <ttl/nn/bits/kernels/activation.hpp>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class softmax<R, host_memory>
{
    const R eps;

  public:
    softmax(R eps) : eps(eps) {}

    void operator()(const tensor_ref<R, 1> &y, const tensor_view<R, 1> &x) const
    {
        for (auto i : range(x.shape().size())) {
            const R tot =
                std::accumulate(x.data(), x.data_end(), static_cast<R>(0),
                                [xi = x.data()[i]](R acc, R xj) {
                                    return acc + std::exp(xj - xi);
                                });
            y.data()[i] = std::max(eps, static_cast<R>(1) / tot);
        }
    }
};

struct relu_scalar {
    template <typename R>
    R operator()(R x)
    {
        return x > 0 ? x : 0.0;
    }
};

template <typename R>
class relu<R, host_memory>
{
  public:
    void operator()(const tensor_ref<R, 1> &y, const tensor_view<R, 1> &x) const
    {
        std::transform(x.data(), x.data_end(), y.data(), relu_scalar());
    }
};
}  // namespace ttl::nn::kernels