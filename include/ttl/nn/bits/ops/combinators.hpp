#pragma once
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::internal
{
template <typename Op>
class batched
{
    const Op op_;

  public:
    batched(const Op &op) : op_(op) {}

    template <rank_t r, typename Dim>
    auto operator()(const ttl::internal::basic_shape<r, Dim> &shape) const
    {
        const auto sub = shape.template subshape<1>();
        return batch(std::get<0>(shape.dims()), op_(sub));
    }

    template <typename R, rank_t r1, rank_t r2, typename D>
    void operator()(const tensor_ref<R, r1, D> &y,
                    const tensor_view<R, r2, D> &x) const
    {
        // FIXME: optimize for cuda tensor
        for (auto i : range<0>(x)) { op_(y[i], x[i]); }
    }
};

template <typename Op>
batched<Op> make_batched(const Op &op)
{
    return batched(op);
}
}  // namespace ttl::nn::ops::internal
