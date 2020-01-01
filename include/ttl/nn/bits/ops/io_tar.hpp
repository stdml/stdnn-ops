#pragma once
#include <cstdint>

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace ttl::nn::ops::internal
{
struct tar_header {
    std::string filename;
    std::int64_t mode;
    std::int64_t uid;
    std::int64_t gid;
    std::string size_oct;

    // char mod_time[12];
    // char checksum[8];
    // char link_indicator;
    // char linked_filename;

    // derived fields
    std::int64_t size;
    std::int64_t data_offset;
};

// https://en.wikipedia.org/wiki/Tar_(computing)#Header
inline void parse_tar_header(tar_header &th, const char *buffer)
{
    th.filename = std::string(buffer, 100).c_str();
    th.mode = *reinterpret_cast<const int64_t *>(buffer + 100);
    th.uid = *reinterpret_cast<const int64_t *>(buffer + 108);
    th.gid = *reinterpret_cast<const int64_t *>(buffer + 116);
    th.size_oct = std::string(buffer + 124, 12);

    std::size_t pos;
    th.size = std::stoll(th.size_oct, &pos, 8);
}

class tar_index
{
  public:
    using index_t = std::map<std::string, tar_header>;

    tar_index(const std::string &filename, const index_t &index)
        : filename_(filename), index_(index)
    {
    }

    auto operator()(const std::string &name) const
    {
        if (index_.count(name) == 0) {
            throw std::invalid_argument("file not found: " + name);
        }
        return index_.at(name);
    }

  private:
    const std::string filename_;
    const index_t index_;
};

inline tar_index make_tar_index(const std::string &filename)
{
    constexpr int block_size = 512;

    std::ifstream fs(filename, std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("can't open file: " + filename);
    }

    const auto begin = fs.tellg();
    const auto fsize = [&fs, begin] {
        fs.seekg(0, std::ios::end);
        const auto fsize = fs.tellg() - begin;
        fs.seekg(0, std::ios::beg);
        return fsize;
    }();

    if (fsize % block_size) {
        throw std::runtime_error("tar file size must be a multiple of " +
                                 std::to_string(block_size));
    }

    char buffer[block_size];
    tar_index::index_t index;
    while (fs.tellg() - begin < fsize) {
        fs.read(buffer, block_size);
        if (buffer[0] == 0) { break; }

        tar_header th;
        parse_tar_header(th, buffer);
        th.data_offset = fs.tellg() - begin;
        index[th.filename] = th;

        const int n_blocks =
            th.size / block_size + (th.size % block_size ? 1 : 0);
        fs.seekg(n_blocks * block_size, std::ios::cur);
    }

    return tar_index(filename, index);
}
}  // namespace ttl::nn::ops::internal
