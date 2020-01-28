#pragma once
#include <cmath>

#include <numeric>

#include <ttl/device>
#include <ttl/nn/bits/kernels/activation.hpp>
#include <ttl/nn/bits/kernels/cpu/elementary.hpp>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class softmax<host_memory, R>
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

    void operator()(const tensor_ref<R, 2> &y, const tensor_view<R, 2> &x) const
    {
        for (auto i : range<0>(x)) { operator()(y[i], x[i]); }
    }
};

struct host_scalar_relu {
    template <typename R>
    R operator()(R x)
    {
        return x > 0 ? x : static_cast<R>(0);
    }
};

template <typename R>
class relu<host_memory, R> : public host_pointwise<host_scalar_relu, R>
{
};

struct host_scaler_relu_grad {
    template <typename R>
    R operator()(R gy, R x)
    {
        return gy * (x > 0 ? static_cast<R>(1) : static_cast<R>(0));
    }
};

template <typename R>
class relu_grad<host_memory, R>
    : public host_binary_pointwise<host_scaler_relu_grad, R>
{
};
}  // namespace ttl::nn::kernels
