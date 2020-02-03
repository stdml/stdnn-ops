#pragma once
#include <array>

#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
namespace internal
{
template <ttl::rank_t p, ttl::rank_t r>
shape<r> concat2shapes(const shape<r> &s, const shape<r> &t)
{
    using dim_t = typename shape<r>::dimension_type;
    std::array<dim_t, r> dims;
    for (auto i : range(r)) {
        if (i == p) {
            dims[i] = s.dims()[i] + t.dims()[i];
        } else {
            contract_assert(s.dims()[i] == t.dims()[i]);
            dims[i] = s.dims()[i];
        }
    }
    return shape<r>(dims);
}

template <ttl::rank_t p, ttl::rank_t r>
shape<r> concat_shape(const shape<r> &s)
{
    return s;
}

template <ttl::rank_t p, ttl::rank_t r, typename... S>
shape<r> concat_shape(const shape<r> &s, const S &... ss)
{
    return concat2shapes<p, r>(s, concat_shape<p, r>(ss...));
}
}  // namespace internal

template <typename F, typename Y, typename... Xs>
void check_shape(const F &f, const Y &y, const Xs &... xs)
{
    if (y.shape() != f(xs.shape()...)) {
        throw std::logic_error("shape check failed");
    }
}

template <arity_t i, typename Op, ttl::rank_t ry, ttl::rank_t... rx>
auto gradient_shape(const Op &infer, const shape<ry> &gy, const shape<ry> &y,
                    const shape<rx> &... xs)
{
    contract_assert_eq(y, infer(xs...));
    contract_assert_eq(y, gy);
    return std::get<i>(std::make_tuple(xs...));
};
}  // namespace ttl::nn::ops
