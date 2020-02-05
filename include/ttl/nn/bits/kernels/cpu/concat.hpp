#pragma once
#include <ttl/algorithm>
#include <ttl/device>
#include <ttl/nn/bits/kernels/concat.hpp>
#include <ttl/nn/traits>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class concat_channel4d<host_memory, traits::nchw, 2, R>
{
  public:
    void operator()(const tensor_ref<R, 4> &t, const tensor_view<R, 4> &x,
                    const tensor_view<R, 4> &y) const
    {
        auto [n, c_all, h, w] = t.dims();
        // TODO: check equality
        auto [_1n, c_1, _1h, _1w] = x.dims();
        // auto [_2n, c_2, _2h, _2w] = y.dims();

        const auto c1 = c_1;
        // const auto c2 = c_2;

        for (auto l : range(n)) {
            for (auto k : range(c_all)) {
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        t.at(l, k, i, j) = [&] {
                            if (k < c1) {
                                return x.at(l, k, i, j);
                            } else {
                                return y.at(l, k - c1, i, j);
                            }
                        }();
                    }
                }
            }
        }
    }
};

template <typename R>
class concat_channel4d<host_memory, traits::nhwc, 3, R>
{
  public:
    void operator()(const tensor_ref<R, 4> &t, const tensor_view<R, 4> &x,
                    const tensor_view<R, 4> &y,
                    const tensor_view<R, 4> &z) const
    {
        auto [n, h, w, c_all] = t.dims();
        // TODO: check equality
        auto [_1n, _1h, _1w, c_1] = x.dims();
        auto [_2n, _2h, _2w, c_2] = y.dims();
        // auto [_3n, _3h, _3w, c_3] = z.dims();

        const auto c1 = c_1;
        const auto c2 = c_2;
        // const auto c3 = c_3;

        for (auto l : range(n)) {
            for (auto i : range(h)) {
                for (auto j : range(w)) {
                    for (auto k : range(c_all)) {
                        t.at(l, i, j, k) = [&] {
                            if (k < c1) {
                                return x.at(l, i, j, k);
                            } else if (k < c1 + c2) {
                                return y.at(l, i, j, k - c1);
                            } else {
                                return z.at(l, i, j, k - c1 - c2);
                            }
                        }();
                    }
                }
            }
        }
    }
};
}  // namespace ttl::nn::kernels
