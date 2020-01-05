#pragma once
#include <ttl/nn/bits/kernels/loss.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>

namespace ttl::nn::ops
{
class xentropy : public binary_reduce_function
{
  public:
    using binary_reduce_function::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r - 1, D> &z,
                    const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        kernels::xentropy<D, R>()(flatten(z), as_matrix<r - 1, 1>(x),
                                  as_matrix<r - 1, 1>(y));
    }
};
}  // namespace ttl::nn::ops
