#pragma once
#include <ttl/nn/bits/kernels/cpu/hash.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename N>
class basic_crc
{
    N poly_;

  public:
    basic_crc(const N poly) : poly_(poly) {}

    template <rank_t r>
    shape<0> operator()(const shape<r> &) const
    {
        return shape<0>();
    }

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<N, 0, D> &y,
                    const tensor_view<R, r, D> &x) const
    {
        kernels::crc<D, N> kernel(poly_);  // FIXME : cache kernel
        kernel(y, x);
    }
};

template <typename N = uint32_t>
class crc : basic_crc<N>
{
    using P = basic_crc<N>;
    using P::P;

  public:
    using P::operator();
};

template <typename N, N seed>
class standard_crc : basic_crc<N>
{
    using P = basic_crc<N>;

  public:
    standard_crc() : P(seed) {}

    using P::operator();
};

struct crc_polynomials {
    static constexpr uint16_t usb = static_cast<uint16_t>(0xa001);
    static constexpr uint32_t ieee = static_cast<uint32_t>(0xedb88320);
    static constexpr uint64_t ecma = static_cast<uint64_t>(0xC96C5795D7870F42);
};

using crc16_usb = standard_crc<uint16_t, crc_polynomials::usb>;
using crc32_ieee = standard_crc<uint32_t, crc_polynomials::ieee>;
using crc64_ecma = standard_crc<uint64_t, crc_polynomials::ecma>;
}  // namespace ttl::nn::ops
