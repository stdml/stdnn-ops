#pragma once
#include <ttl/nn/bits/engines/linag.hpp>
#include <ttl/nn/bits/ops/activation.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <arity_t>
class softmax;

template <>
class softmax<0>
{
  public:
    template <rank_t r>
    shape<r> operator()(const shape<r> &gy, const shape<r> &y,
                        const shape<r> &x) const
    {
        return ttl::nn::ops::gradient_shape<0>(ttl::nn::ops::softmax(), gy, y,
                                               x);
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 1, D> &gx,
                    const tensor_view<R, 1, D> &gy,
                    const tensor_view<R, 1, D> &y,
                    const tensor_view<R, 1, D> &x) const
    {
        const auto n = x.size();
        const tensor<R, 2, D> g(n, n);
        for (auto i : range(n)) {
            for (auto j : range(n)) {
                if (i == j) {
                    const R v = y.at(i);
                    g.at(i, j) = v * (static_cast<R>(1) - v);
                } else {
                    g.at(i, j) = -y.at(i) * y.at(j);
                }
            }
        }
        using E = typename engines::default_blas<D>::type;
        nn::engines::linag<E>::vm(gy, view(g), gx);
    }

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 2, D> &gx,
                    const tensor_view<R, 2, D> &gy,
                    const tensor_view<R, 2, D> &y,
                    const tensor_view<R, 2, D> &x) const
    {
        for (auto i : range<0>(x)) { operator()(gx[i], gy[i], y[i], x[i]); }
    }
};
}  // namespace ttl::nn::ops::grad
