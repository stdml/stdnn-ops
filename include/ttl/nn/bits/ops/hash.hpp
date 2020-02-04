#pragma once
#include <ttl/nn/bits/kernels/cpu/hash.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
namespace internal
{
template <typename N>
N crc_seed;

template <>
constexpr uint32_t crc_seed<uint32_t> =
    kernels::crc<host_memory, uint32_t>::IEEE;

template <>
constexpr uint64_t crc_seed<uint64_t> =
    kernels::crc<host_memory, uint64_t>::ECMA;
}  // namespace internal

template <typename N = uint32_t>
class crc
{
    N poly_;

  public:
    crc(const N poly = internal::crc_seed<N>) : poly_(poly) {}

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
