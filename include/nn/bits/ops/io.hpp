#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>

#include <experimental/contract>

#include <bits/std_scalar_type_encoding.hpp>

#include <nn/bits/ops/io_tar.hpp>
#include <nn/common.hpp>

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

using encoding = ttl::internal::idx_format::encoding;

template <typename R, ttl::rank_t r>
void read(const ttl::tensor_ref<R, r> &y, std::istream &is)
{
    {
        char magic[4];
        is.read(magic, 4);
        contract_assert(magic[0] == 0);
        contract_assert(magic[1] == 0);
        const uint8_t type = magic[2];
        const uint8_t rank = magic[3];
        contract_assert(type == encoding::value<R>());
        contract_assert(rank == r);
    }
    {
        uint32_t dims[r];
        is.read(reinterpret_cast<char *>(dims), r * sizeof(uint32_t));
        for (auto i : range(r)) {
            swap_byte_endian(dims[i]);
            contract_assert_eq(dims[i], y.shape().dims[i]);
        }
    }
    is.read(reinterpret_cast<char *>(y.data()), sizeof(R) * y.shape().size());
}

template <typename R, ttl::rank_t r>
void write(const ttl::tensor_view<R, r> &x, std::ostream &os)
{
    {
        char magic[4] = {0, 0, encoding::value<R>(), r};
        os.write(magic, 4);
    }
    {
        uint32_t dims[r];
        for (auto i : range(r)) {
            dims[i] = x.shape().dims[i];
            swap_byte_endian(dims[i]);
        }
        os.write(reinterpret_cast<const char *>(dims), sizeof(uint32_t) * r);
    }
    os.write(reinterpret_cast<const char *>(x.data()),
             sizeof(R) * x.shape().size());
}

}  // namespace internal::idx_format

class readfile
{
    const std::string filename_;

  public:
    readfile(const std::string &filename) : filename_(filename) {}

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y) const
    {
        std::ifstream fs(filename_, std::ios::binary);
        internal::idx_format::read(y, fs);
    }
};

class writefile
{
    const std::string filename_;

  public:
    writefile(const std::string &filename) : filename_(filename) {}

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_view<R, r> &x) const
    {
        std::ofstream fs(filename_, std::ios::binary);
        internal::idx_format::write(x, fs);
    }
};

// read a tensor from a tar file of idx files
class readtar
{
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
        const auto idx = nn::ops::internal::make_tar_index(filename_);
        const auto info = idx(name_);
        std::ifstream fs(filename_, std::ios::binary);
        fs.seekg(info.data_offset, std::ios::beg);
        internal::idx_format::read(y, fs);
    }
};

}  // namespace nn::ops
