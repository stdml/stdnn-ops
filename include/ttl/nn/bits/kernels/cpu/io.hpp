#pragma once
#include <cstdint>

#include <fstream>
#include <string>

#include <ttl/bits/idx_encoding.hpp>
#include <ttl/device>
#include <ttl/nn/bits/kernels/io.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
namespace internal
{
inline uint32_t byte_endian_swapped(uint32_t x)
{
    uint8_t *p = reinterpret_cast<uint8_t *>(&x);
    std::swap(p[0], p[3]);
    std::swap(p[1], p[2]);
    return x;
}
}  // namespace internal

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

template <typename encoding>
class serializer
{
  public:
    template <typename R, rank_t r>
    static void read(const tensor_ref<R, r> &y, std::istream &is)
    {
        {
            char magic[4];
            is.read(magic, 4);
            contract_assert(magic[0] == 0);
            contract_assert(magic[1] == 0);
            const uint8_t type = magic[2];
            const uint8_t rank = magic[3];
            contract_assert(type == encoding::template value<R>());
            contract_assert(rank == r);
        }
        {
            uint32_t buffer[r];
            is.read(reinterpret_cast<char *>(buffer), r * sizeof(uint32_t));
            std::transform(buffer, buffer + r, buffer, byte_endian_swapped);
            for (auto i : range(r)) {
                contract_assert_eq(buffer[i], y.shape().dims()[i]);
            }
        }
        is.read(reinterpret_cast<char *>(y.data()),
                sizeof(R) * y.shape().size());
    }

    template <typename R, rank_t r>
    static void write(const tensor_view<R, r> &x, std::ostream &os)
    {
        {
            char magic[4] = {0, 0, encoding::template value<R>(), r};
            os.write(magic, 4);
        }
        {
            uint32_t buffer[r];
            const auto dims = x.shape().dims();
            std::transform(dims.begin(), dims.end(), buffer,
                           byte_endian_swapped);
            os.write(reinterpret_cast<const char *>(buffer),
                     sizeof(uint32_t) * r);
        }
        os.write(reinterpret_cast<const char *>(x.data()),
                 sizeof(R) * x.shape().size());
    }
};
}  // namespace internal::idx_format

template <typename R>
class readfile<host_memory, R>
{
    using serializer =
        internal::idx_format::serializer<ttl::internal::idx_format::encoding>;

  public:
    template <rank_t r>
    void operator()(const tensor_ref<R, r> &y,
                    const tensor_view<std::string, 0> &x) const
    {
        const std::string &filename = x;
        std::ifstream fs(filename, std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("can't open " + filename);
        }
        serializer::read(y, fs);
    }
};

template <typename R>
class writefile<host_memory, R>
{
    using serializer =
        internal::idx_format::serializer<ttl::internal::idx_format::encoding>;

  public:
    template <rank_t r>
    void operator()(const tensor_view<R, r> &x,
                    const tensor_view<std::string, 0> &y) const
    {
        const std::string &filename = y;
        std::ofstream fs(filename, std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("can't open " + filename);
        }
        serializer::write(x, fs);
    }
};
}  // namespace ttl::nn::kernels
