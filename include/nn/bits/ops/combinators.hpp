#pragma once
#include <experimental/contract>
#include <nn/bits/ops/shape_algo.hpp>

namespace nn::ops::internal
{
template <typename Op> class batched
{
    const Op op_;

  public:
    batched(const Op &op) : op_(op) {}

    template <ttl::rank_t r> auto operator()(const shape<r> &shp) const
    {
        const auto sub = shp.template subshape<1>();
        return batch(std::get<0>(shp.dims), op_(sub));
    }

    template <typename R, ttl::rank_t r1, ttl::rank_t r2>
    void operator()(const ttl::tensor_ref<R, r1> &y,
                    const ttl::tensor_view<R, r2> &x) const
    {
        const auto n = std::get<0>(x.shape().dims);
        contract_assert(n == std::get<0>(y.shape().dims));
        for (auto i : range(n)) { op_(y[i], x[i]); }
    }
};

template <typename Op> batched<Op> make_batched(const Op &op)
{
    return batched(op);
}
}  // namespace nn::ops::internal
