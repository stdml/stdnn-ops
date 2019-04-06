#pragma once
#include <array>

#include <nn/bits/ops/shape_algo.hpp>
#include <nn/bits/ops/traits.hpp>
#include <nn/common.hpp>

namespace nn::ops
{

// TODO: make it more generic
template <typename image_order, uint8_t arity> class concat_channel4d_impl;

template <> class concat_channel4d_impl<nhwc, 3>
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y,
                        const shape<4> &z) const
    {
        return internal::concat_shape<channel_position<nn::ops::nhwc>>(x, y, z);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &t,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y,
                    const ttl::tensor_view<R, 4> &z) const
    {
        auto [n, h, w, c_all] = t.shape().dims;
        // TODO: check equality
        auto [_1n, _1h, _1w, c_1] = x.shape().dims;
        auto [_2n, _2h, _2w, c_2] = y.shape().dims;
        auto [_3n, _3h, _3w, c_3] = z.shape().dims;

        const auto c1 = c_1;
        const auto c2 = c_2;
        const auto c3 = c_3;

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

template <> class concat_channel4d_impl<nchw, 2>
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        return internal::concat_shape<channel_position<nn::ops::nchw>>(x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &t,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
        auto [n, c_all, h, w] = t.shape().dims;
        // TODO: check equality
        auto [_1n, c_1, _1h, _1w] = x.shape().dims;
        auto [_2n, c_2, _2h, _2w] = y.shape().dims;

        const auto c1 = c_1;
        const auto c2 = c_2;

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

template <typename image_order> class concat_channel4d
{
  public:
    template <typename... S> shape<4> operator()(const S &... s) const
    {
        return concat_channel4d_impl<image_order, sizeof...(S)>()(s...);
    }

    template <typename R, typename... T>
    void operator()(const ttl::tensor_ref<R, 4> &t, const T &... x) const
    {
        return concat_channel4d_impl<image_order, sizeof...(T)>()(t, x...);
    }
};

}  // namespace nn::ops
