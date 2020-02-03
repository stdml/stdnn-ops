#pragma once
#include <type_traits>

#include <ttl/device>
#include <ttl/nn/bits/kernels/hash.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename N>
class crc<host_memory, N>
{
    static_assert(std::is_unsigned<N>::value, "");

    N table[256];

    N sum(const uint8_t *s, const uint8_t *t,
          const N init = static_cast<N>(-1)) const
    {
        return std::accumulate(s, t, init, [&](N a, uint8_t b) {
            return (a >> 8) ^ table[static_cast<uint8_t>(a ^ b)];
        });
    }

    N t(const N s, const N poly)
    {
        return (s & 1) == 1 ? (s >> 1) ^ poly : s >> 1;
    }

  public:
    static constexpr N IEEE = static_cast<uint32_t>(0xedb88320);
    static constexpr N ECMA = static_cast<uint64_t>(0xC96C5795D7870F42);

    crc(const N poly = IEEE)
    {
        for (uint32_t i = 0; i < 256; i++) {
            N crc = i;
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            table[i] = crc;
        }
    }

    template <typename R, rank_t r>
    void operator()(const tensor_ref<N, 0> &y, const tensor_view<R, r> &x) const
    {
        y.data()[0] = sum(reinterpret_cast<const uint8_t *>(x.data()),
                          reinterpret_cast<const uint8_t *>(x.data_end())) ^
                      static_cast<N>(-1);
    }
};
}  // namespace ttl::nn::kernels
