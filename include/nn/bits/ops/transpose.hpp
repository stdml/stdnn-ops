#pragma once
#include <array>
#include <functional>

#include <nn/common.hpp>

namespace nn::ops
{
// TODO: use std::identity since c++20
struct std_identity {
    template <typename R> R operator()(const R &x) const { return x; }
};

template <typename to_image_order, typename from_image_order> class transpose;

template <> class transpose<nchw, nhwc>
{
  public:
    shape<4> operator()(const shape<4> &x) const
    {
        const auto [n, h, w, c] = x.dims;
        return shape<4>(n, c, h, w);
    }

    template <typename R, typename S, typename F = std_identity>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<S, 4> &x, const F &f = F()) const
    {
        const auto [n, h, w, c] = x.shape().dims;

        for (auto l : range(n)) {
            for (auto k : range(c)) {
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        y.at(l, k, i, j) = f(x.at(l, i, j, k));
                    }
                }
            }
        }
    }
};

template <> class transpose<nhwc, nchw>
{
  public:
    shape<4> operator()(const shape<4> &x) const
    {
        const auto [n, c, h, w] = x.dims;
        return shape<4>(n, h, w, c);
    }

    template <typename R, typename S, typename F = std_identity>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<S, 4> &x, const F &f = F()) const
    {
        const auto [n, c, h, w] = x.shape().dims;

        for (auto l : range(n)) {
            for (auto i : range(h)) {
                for (auto j : range(w)) {
                    for (auto k : range(c)) {
                        y.at(l, i, j, k) = f(x.at(l, k, i, j));
                    }
                }
            }
        }
    }
};

using to_channels_first = transpose<nchw, nhwc>;
using to_channels_last = transpose<nhwc, nchw>;

}  // namespace nn::ops
