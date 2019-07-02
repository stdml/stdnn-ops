#pragma once
#include <ttl/algorithm>

#include <nn/bits/ops/reshape.hpp>
#include <nn/bits/ops/shape_algo.hpp>
#include <nn/bits/ops/summary.hpp>
#include <nn/common.hpp>

namespace nn::ops
{
class mean : public nn::ops::reduce_function
{
  public:
    using reduce_function::operator();

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<R, r + 1> &x) const
    {
        const auto x_flat = nn::ops::as_matrix<r, 1, ttl::tensor_view<R, 2>>(x);
        for (auto i : range(y.shape().size())) {
            // TODO: add mean to ttl/algorithm
            y.data()[i] = nn::ops::summaries::mean()(view(x_flat[i]));
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
    for (const auto xi : x) { nn::ops::add()(y, view(y), xi); }
}

}  // namespace internal
}  // namespace nn::ops
