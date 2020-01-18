#pragma once
#include <ttl/algorithm>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/bits/ops/summary.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class mean : public ttl::nn::ops::reduce_function
{
  public:
    using reduce_function::operator();

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<R, r + 1> &x) const
    {
        const auto x_flat = ttl::nn::ops::as_matrix<r, 1>(x);
        for (auto i : range<0>(x_flat)) {
            // TODO: add mean to ttl/algorithm
            y.data()[i] = ttl::nn::ops::summaries::mean()(x_flat[i]);
        }
    }
};

namespace internal
{

template <typename R>  // Y[i] = \sum X[i, j]
void inner_contraction(const ttl::tensor_ref<R, 1> &y,
                       const ttl::tensor_view<R, 2> &x)
{
    std::transform(x.begin(), x.end(), y.begin(),
                   // ttl::sum // FIXME: use ttl::sum directly
                   [](const ttl::tensor_view<R, 1> &v) { return ttl::sum(v); });
}

template <typename R>  // Y[j] = \sum X[i, j]
void outter_contraction(const ttl::tensor_ref<R, 1> &y,
                        const ttl::tensor_view<R, 2> &x)
{
    ttl::fill(y, static_cast<R>(0));
    for (const auto xi : x) { ttl::nn::ops::add()(y, view(y), xi); }
}

}  // namespace internal
}  // namespace ttl::nn::ops
