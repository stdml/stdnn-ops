#pragma once
#include <ttl/algorithm>

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
}  // namespace nn::ops
