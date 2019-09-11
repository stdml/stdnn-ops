#pragma once
#include <nn/bits/ops/shape_algo.hpp>

namespace nn::ops
{
template <ttl::rank_t p, ttl::rank_t q, typename R>
ttl::tensor_ref<R, 2> as_matrix(const ttl::tensor_ref<R, p + q> &t)
{
    return ttl::tensor_ref<R, 2>(
        t.data(), ttl::internal::flatten_shape<p, q>()(t.shape()));
}

template <ttl::rank_t p, ttl::rank_t q, typename R>
ttl::tensor_view<R, 2> as_matrix(const ttl::tensor_view<R, p + q> &t)
{
    return ttl::tensor_view<R, 2>(
        t.data(), ttl::internal::flatten_shape<p, q>()(t.shape()));
}

template <ttl::rank_t... rs> class copy_flatten
{
    static constexpr ttl::rank_t in_rank =
        ttl::internal::int_seq_sum<ttl::rank_t, rs...>::value;
    static constexpr ttl::rank_t out_rank = sizeof...(rs);

  public:
    ttl::shape<out_rank> operator()(const ttl::shape<in_rank> &x) const
    {
        return ttl::internal::flatten_shape<rs...>()(x);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, out_rank> &y,
                    const ttl::tensor_view<R, in_rank> &x) const
    {
        std::copy(x.data(), x.data_end(), y.data());
    }
};
}  // namespace nn::ops
