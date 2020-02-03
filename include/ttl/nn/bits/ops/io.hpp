#pragma once
#include <cstdint>

#include <algorithm>
#include <fstream>

#include <ttl/bits/idx_encoding.hpp>
#include <ttl/nn/bits/ops/io_tar.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
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

template <typename encoding = ttl::internal::idx_format::encoding>
class serializer
{
  public:
    template <typename R, ttl::rank_t r>
    static void read(const ttl::tensor_ref<R, r> &y, std::istream &is)
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

    template <typename R, ttl::rank_t r>
    static void write(const ttl::tensor_view<R, r> &x, std::ostream &os)
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

class readfile
{
    using serializer = internal::idx_format::serializer<>;

    const std::string filename_;

  public:
    readfile(const std::string &filename) : filename_(filename) {}

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y) const
    {
        std::ifstream fs(filename_, std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("can't open " + filename_);
        }
        serializer::read(y, fs);
    }
};

class writefile
{
    using serializer = internal::idx_format::serializer<>;

    const std::string filename_;

  public:
    writefile(const std::string &filename) : filename_(filename) {}

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_view<R, r> &x) const
    {
        std::ofstream fs(filename_, std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("can't open " + filename_);
        }
        serializer::write(x, fs);
    }
};

// read a tensor from a tar file of idx files
class readtar
{
    using serializer = internal::idx_format::serializer<>;

    const std::string filename_;
    const std::string name_;

  public:
    readtar(const std::string &filename, const std::string &name)
        : filename_(filename), name_(name)
    {
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y) const
    {
        const auto idx = ttl::nn::ops::internal::make_tar_index(filename_);
        const auto info = idx(name_);
        std::ifstream fs(filename_, std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("can't open " + filename_);
        }
        fs.seekg(info.data_offset, std::ios::beg);
        serializer::read(y, fs);
    }
};
}  // namespace ttl::nn::ops
