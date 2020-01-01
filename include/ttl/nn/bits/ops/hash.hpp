#pragma once
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename N = uint32_t> class crc
{
    N table[256];

    N sum(const uint8_t *s, const uint8_t *t, const N init = 0xffffffff) const
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
    static constexpr N IEEE = 0xedb88320;

    crc(const N poly = IEEE)
    {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
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

    template <ttl::rank_t r>
    shape<0> operator()(const shape<r> &x, const shape<1> &y) const
    {
        return shape<0>();
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<N, 0> &y,
                    const ttl::tensor_view<R, r> &x) const
    {
        y.data()[0] = sum(reinterpret_cast<const uint8_t *>(x.data()),
                          reinterpret_cast<const uint8_t *>(x.data_end())) ^
                      0xffffffff;
    }
};
}  // namespace ttl::nn::ops
