#pragma once
#include <ttl/nn/bits/kernels/cpu/io.hpp>
#include <ttl/nn/bits/ops/io_tar.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class readfile
{
    const std::string filename_;

  public:
    readfile(const std::string &filename) : filename_(filename) {}

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &y) const
    {
        const tensor_view<std::string, 0> x(&filename_);
        kernels::readfile<D, R>()(y, x);
    }
};

class writefile
{
    const std::string filename_;

  public:
    writefile(const std::string &filename) : filename_(filename) {}

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_view<R, r, D> &x) const
    {
        const tensor_view<std::string, 0> y(&filename_);
        kernels::writefile<D, R>()(x, y);
    }
};

// read a tensor from a tar file of idx files
class readtar
{
    using serializer = kernels::internal::idx_format::serializer<
        ttl::internal::idx_format::encoding>;

    const std::string filename_;
    const std::string name_;

  public:
    readtar(const std::string &filename, const std::string &name)
        : filename_(filename), name_(name)
    {
    }

    template <typename R, rank_t r>
    void operator()(const tensor_ref<R, r> &y) const
    {
        const auto idx = ops::internal::make_tar_index(filename_);
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
