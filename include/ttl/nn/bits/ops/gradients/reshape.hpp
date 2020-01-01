#pragma once
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <ttl::rank_t... rs> class copy_flatten
{
    static constexpr ttl::rank_t in_rank =
        ttl::internal::int_seq_sum<ttl::rank_t, rs...>::value;
    static constexpr ttl::rank_t out_rank = sizeof...(rs);

  public:
    shape<in_rank> operator()(const shape<out_rank> &gy,
                              const shape<out_rank> &y,
                              const shape<in_rank> &x) const
    {
        return ttl::nn::ops::gradient_shape<0>(
            ttl::nn::ops::copy_flatten<rs...>(), gy, y, x);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, in_rank> &gx,
                    const ttl::tensor_view<R, out_rank> &gy,
                    const ttl::tensor_view<R, out_rank> &_y,
                    const ttl::tensor_view<R, in_rank> &_x) const
    {
        std::copy(gy.data(), gy.data_end(), gx.data());
    }
};
}  // namespace ttl::nn::ops::grad
