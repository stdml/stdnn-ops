#pragma once
#include <algorithm>

#include <experimental/contract>
#include <nn/bits/ops/traits.hpp>
#include <stdtensor>

namespace nn::ops
{

template <typename TensorOrder, typename Op> class apply_bias;

template <typename Op> class apply_bias<hw, Op>
{
  public:
    shape<2> operator()(const shape<2> &x, const shape<1> &y) const
    {
        contract_assert(x.dims[1] == y.dims[0]);
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        const auto [h, w] = x.shape().dims;
        for (auto i : range(h)) {
            for (auto j : range(w)) { z.at(i, j) = Op()(x.at(i, j), y.at(j)); }
        }
    }
};

template <typename Op> class apply_bias<nhwc, Op>
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &y) const
    {
        contract_assert(ops::channel_size<nhwc>(x) == y.dims[0]);
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        const auto [n, h, w, c] = x.shape().dims;
        for (auto b : range(n)) {
            for (auto i : range(h)) {
                for (auto j : range(w)) {
                    for (auto l : range(c)) {
                        z.at(b, i, j, l) = Op()(x.at(b, i, j, l), y.at(l));
                    }
                }
            }
        }
    }
};

template <typename Op> class apply_bias<nchw, Op>
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &y) const
    {
        contract_assert(ops::channel_size<nchw>(x) == y.dims[0]);
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        const auto [n, c, h, w] = x.shape().dims;
        for (auto b : range(n)) {
            for (auto l : range(c)) {
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        z.at(b, l, i, j) = Op()(x.at(b, l, i, j), y.at(l));
                    }
                }
            }
        }
    }
};

}  // namespace nn::ops
