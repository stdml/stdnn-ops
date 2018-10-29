#pragma once
#include <algorithm>
#include <cmath>

#include <experimental/contract>
#include <stdtensor>

namespace nn::ops
{
namespace internal
{

template <typename R> struct softmax {
    const size_t size;

    const R eps;

    softmax(size_t size, R eps) : size(size), eps(eps) {}

    void operator()(R *output, const R *input) const
    {
        for (auto i : range(size)) {
            const R tot =
                std::accumulate(input, input + size, static_cast<R>(0),
                                [xi = input[i]](R acc, R xj) {
                                    return acc + std::exp(xi - xj);
                                });
            output[i] = std::max(eps, static_cast<R>(1) / tot);
        }
    }
};

}  // namespace internal

template <ttl::rank_t r, typename R> class softmax_impl;

template <typename R> class softmax_impl<1, R>
{
    const R eps_;

  public:
    softmax_impl(R eps) : eps_(eps) {}

    void operator()(const ttl::tensor_ref<R, 1> &y,
                    const ttl::tensor_view<R, 1> &x) const
    {
        (internal::softmax<R>(x.shape().size(), eps_))(y.data(), x.data());
    }
};

template <typename R> class softmax_impl<2, R>
{
    const R eps_;

  public:
    softmax_impl(R eps) : eps_(eps) {}

    void operator()(const ttl::tensor_ref<R, 2> &y,
                    const ttl::tensor_view<R, 2> &x) const
    {
        const auto [n, k] = x.shape().dims;
        const auto op = internal::softmax<R>(k, eps_);
        for (auto i : range(n)) { op(y[i].data(), x[i].data()); }
    }
};

class softmax
{
  public:
    template <ttl::rank_t r> shape<r> operator()(const shape<r> &x) const
    {
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<R, r> &x,
                    R eps = static_cast<R>(1e-6)) const
    {
        (softmax_impl<r, R>(eps))(y, x);
    }
};
}  // namespace nn::ops
