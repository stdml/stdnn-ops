#pragma once
#include <ttl/nn/bits/ops/shape_algo.hpp>

namespace ttl::nn::ops
{
template <ttl::rank_t p, rank_t q, typename R, typename D>
tensor_ref<R, 2, D> as_matrix(const tensor_ref<R, p + q, D> &t)
{
    return tensor_ref<R, 2, D>(t.data(),
                               ttl::internal::flatten_shape<p, q>()(t.shape()));
}

template <rank_t p, rank_t q, typename R, typename D>
tensor_view<R, 2, D> as_matrix(const tensor_view<R, p + q, D> &t)
{
    return tensor_view<R, 2, D>(
        t.data(), ttl::internal::flatten_shape<p, q>()(t.shape()));
}

template <rank_t... rs>
class copy_flatten
{
    static constexpr rank_t in_rank =
        ttl::internal::int_seq_sum<rank_t, rs...>::value;
    static constexpr rank_t out_rank = sizeof...(rs);

  public:
    ttl::shape<out_rank> operator()(const ttl::shape<in_rank> &x) const
    {
        return ttl::internal::flatten_shape<rs...>()(x);
    }

    template <typename R>
    void operator()(const tensor_ref<R, out_rank> &y,
                    const tensor_view<R, in_rank> &x) const
    {
        std::copy(x.data(), x.data_end(), y.data());
    }
};
}  // namespace ttl::nn::ops
