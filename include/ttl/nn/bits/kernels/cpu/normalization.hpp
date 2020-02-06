#pragma once
#include <cmath>

#include <ttl/device>
#include <ttl/nn/bits/kernels/normalization.hpp>

namespace ttl::nn::kernels
{
template <typename R>
class normalizer_mean_std
{
    static constexpr R eps = .000001f;

  public:
    R operator()(R x, R mean, R std) const { return (x - mean) / (std + eps); }
};

template <typename R>
class normalizer_mean_var
{
    static constexpr R eps = .000001f;

  public:
    R operator()(R x, R mean, R var) const
    {
        return (x - mean) / (std::sqrt(var) + eps);
    }
};

template <typename R>
class batch_norm<host_memory, traits::nhwc, R>
{
  public:
    void operator()(const tensor_ref<R, 4> &y, const tensor_view<R, 4> &x,
                    const tensor_view<R, 1> &rolling_mean,
                    const tensor_view<R, 1> &rolling_var) const
    {
        normalizer_mean_var<R> normalize;
        // FIXME: use apply_bias<traits::nhwc, normalizer_mean_var>
        for (const auto l : range<0>(x)) {
            for (const auto i : range<1>(x)) {
                for (const auto j : range<2>(x)) {
                    for (const auto k : range<3>(x)) {
                        y.at(l, i, j, k) =
                            normalize(x.at(l, i, j, k), rolling_mean.at(k),
                                      rolling_var.at(k));
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
        normalizer_mean_var<R> normalize;
        for (const auto l : range<0>(x)) {
            for (const auto k : range<1>(x)) {
                for (const auto i : range<2>(x)) {
                    for (const auto j : range<3>(x)) {
                        y.at(l, k, i, j) =
                            normalize(x.at(l, k, i, j), rolling_mean.at(k),
                                      rolling_var.at(k));
                    }
                }
            }
        }
    }
};
}  // namespace ttl::nn::kernels
