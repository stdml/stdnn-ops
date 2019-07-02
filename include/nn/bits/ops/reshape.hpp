#pragma once
#include <nn/bits/ops/shape_algo.hpp>

namespace nn::ops
{

template <ttl::rank_t p, ttl::rank_t q, typename R>
ttl::tensor_ref<R, 2> as_matrix(const ttl::tensor_ref<R, p + q> &t)
{
    return ttl::tensor_ref<R, 2>(t.data(), as_mat_shape<p, q>(t.shape()));
}

template <ttl::rank_t p, ttl::rank_t q, typename R>
ttl::tensor_view<R, 2> as_matrix(const ttl::tensor_view<R, p + q> &t)
{
    return ttl::tensor_view<R, 2>(t.data(), as_mat_shape<p, q>(t.shape()));
}

}  // namespace nn::ops
