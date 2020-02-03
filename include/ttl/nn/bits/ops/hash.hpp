#pragma once
#include <ttl/nn/bits/kernels/cpu/hash.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename N = uint32_t>
class crc
{
    N poly_;

  public:
    crc(const N poly = kernels::crc<host_memory, N>::IEEE) : poly_(poly) {}

    template <rank_t r>
    shape<0> operator()(const shape<r> &) const
    {
        return shape<0>();
    }

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<N, 0, D> &y,
                    const tensor_view<R, r, D> &x) const
    {
        kernels::crc<D, N> kernel(poly_);
        kernel(y, x);
    }
};
}  // namespace ttl::nn::ops
