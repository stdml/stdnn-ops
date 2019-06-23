#pragma once
#include <experimental/contract>
#include <experimental/new_type>
#include <experimental/range>
#include <sstream>

#include <ttl/tensor>

// TODO: commit to upstream
namespace ttl
{
using rank_t = internal::rank_t;

namespace internal
{
template <rank_t r>
bool operator==(const basic_shape<r> &p, const basic_shape<r> &q)
{
    for (auto i : std::experimental::range(r)) {
        if (p.dims[i] != q.dims[i]) { return false; }
    }
    return true;
}

template <rank_t r>
bool operator!=(const basic_shape<r> &p, const basic_shape<r> &q)
{
    for (auto i : std::experimental::range(r)) {
        if (p.dims[i] != q.dims[i]) { return true; }
    }
    return false;
}

}  // namespace internal
}  // namespace ttl

namespace std
{
template <ttl::rank_t r>
string to_string(const ttl::internal::basic_shape<r> &p)
{
    stringstream ss;
    ss << "(";
    for (auto i : experimental::range(r)) {
        if (i > 0) { ss << ","; }
        ss << (int)p.dims[i];
    }
    ss << ")";
    return ss.str();
}

}  // namespace std
namespace nn
{
template <ttl::internal::rank_t r> using shape = ttl::internal::basic_shape<r>;

using std::experimental::range;
}  // namespace nn

namespace nn::experimental::ops
{
using std::experimental::range;
}  // namespace nn::experimental::ops

template <ttl::internal::rank_t r>
inline void contract_assert_eq_(const ttl::internal::basic_shape<r> &x,
                                const ttl::internal::basic_shape<r> &y,
                                const char *file, int line)
{
    if (x != y) {
        throw std::runtime_error(
            "contract_assert_eq failed: " + std::string(file) + ":" +
            std::to_string(line) + ": " + std::to_string(x) +
            " != " + std::to_string(y));
    }
}
