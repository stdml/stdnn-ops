#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>

#include <experimental/contract>
#include <stdtensor>

namespace nn::ops
{
namespace internal::idx_format
{
/*
We adopt the idx format originally defined at
<http://yann.lecun.com/exdb/mnist/> for tensor.

         0        8        16       24       32
         +--------+--------+--------+--------+
         |  zero  |  zero  |  dtype |  rank  |
         +--------+--------+--------+--------+
     MSB |               dim 1 (length)      | LSB
         +--------+--------+--------+--------+
         |               dim 2 (width)       |
         +--------+--------+--------+--------+
         |               dim 3               |
         +--------+--------+--------+--------+
         |               dim 4               |
         +--------+--------+--------+--------+
         |               ...                 |
         +--------+--------+--------+--------+
         |               dim r               |
         +--------+--------+--------+--------+
         |                                   |
         |               data                |
         |                                   |
         +--------+--------+--------+--------+
*/
inline void swap_byte_endian(uint32_t &x)
{
    uint8_t *p = reinterpret_cast<uint8_t *>(&x);
    std::swap(p[0], p[3]);
    std::swap(p[1], p[2]);
}

template <typename> struct idx_type;

template <> struct idx_type<uint8_t> {
    static constexpr uint8_t type = 0x08;
};

template <> struct idx_type<int8_t> {
    static constexpr uint8_t type = 0x09;
};

template <> struct idx_type<int16_t> {
    static constexpr uint8_t type = 0x0B;
};

template <> struct idx_type<int32_t> {
    static constexpr uint8_t type = 0x0C;
};

template <> struct idx_type<float> {
    static constexpr uint8_t type = 0x0D;
};

template <> struct idx_type<double> {
    static constexpr uint8_t type = 0x0E;
};

}  // namespace internal::idx_format

class readfile
{
    std::string filename_;

  public:
    readfile(const std::string &filename) : filename_(filename) {}

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y) const
    {
        std::ifstream fs(filename_, std::ios::binary);
        {
            char magic[4];
            fs.read(magic, 4);
            contract_assert(magic[0] == 0);
            contract_assert(magic[1] == 0);
            const uint8_t type = magic[2];
            const uint8_t rank = magic[3];
            contract_assert(type == internal::idx_format::idx_type<R>::type);
            contract_assert(rank == r);
        }
        {
            uint32_t dims[r];
            fs.read(reinterpret_cast<char *>(dims), r * sizeof(uint32_t));
            for (auto i : range(r)) {
                internal::idx_format::swap_byte_endian(dims[i]);
                contract_assert_eq(dims[i], y.shape().dims[i]);
            }
        }
        fs.read(reinterpret_cast<char *>(y.data()),
                sizeof(R) * y.shape().size());
    }
};
}  // namespace nn::ops
