#pragma once
#include <array>

#include <nn/bits/ops/shape_algo.hpp>

namespace nn::ops
{

template <ttl::rank_t p, ttl::rank_t q, typename T, typename T1>
T as_matrix(const T1 &t)
{
    static_assert(T::rank == 2);
    static_assert(T1::rank == p + q);
    return T(t.data(), as_mat_shape<p, q>(t.shape()));
}

}  // namespace nn::ops
