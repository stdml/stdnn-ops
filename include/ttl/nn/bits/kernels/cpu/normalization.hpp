#pragma once
#include <cmath>

#include <ttl/device>
#include <ttl/nn/bits/kernels/normalization.hpp>

namespace ttl::nn::kernels
{
template <typename R>
class batch_norm<host_memory, traits::nhwc, R>
{
  public:
    void operator()(const tensor_ref<R, 4> &y, const tensor_view<R, 4> &x,
                    const tensor_view<R, 1> &rolling_mean,
                    const tensor_view<R, 1> &rolling_var) const
    {
        constexpr R eps = .000001f;
        for (const auto l : range<0>(x)) {
            for (const auto i : range<1>(x)) {
                for (const auto j : range<2>(x)) {
                    for (const auto k : range<3>(x)) {
                        y.at(l, i, j, k) =
                            (x.at(l, i, j, k) - rolling_mean.at(k)) /
                            (std::sqrt(rolling_var.at(k)) + eps);
                    }
                }
            }
        }
    }
};

template <typename R>
class batch_norm<host_memory, traits::nchw, R>
{
  public:
    void operator()(const tensor_ref<R, 4> &y, const tensor_view<R, 4> &x,
                    const tensor_view<R, 1> &rolling_mean,
                    const tensor_view<R, 1> &rolling_var) const
    {
        constexpr R eps = .000001f;
        for (const auto l : range<0>(x)) {
            for (const auto k : range<1>(x)) {
                for (const auto i : range<2>(x)) {
                    for (const auto j : range<3>(x)) {
                        y.at(l, k, i, j) =
                            (x.at(l, k, i, j) - rolling_mean.at(k)) /
                            (std::sqrt(rolling_var.at(k)) + eps);
                    }
                }
            }
        }
    }
};
}  // namespace ttl::nn::kernels
