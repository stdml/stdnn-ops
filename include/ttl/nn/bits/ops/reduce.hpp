#pragma once
#include <ttl/algorithm>
#include <ttl/nn/bits/ops/reshape.hpp>
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
}  // namespace ttl::nn::ops
