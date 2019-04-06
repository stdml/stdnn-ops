#pragma once
#include <stdtensor>

#include <experimental/contract>
#include <experimental/new_type>
#include <experimental/range>

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

}  // namespace internal
}  // namespace ttl

namespace nn
{
template <ttl::internal::rank_t r> using shape = ttl::internal::basic_shape<r>;
}

namespace nn::engines
{
using std::experimental::range;
}

namespace nn::ops
{
using std::experimental::range;
}

namespace nn::layers
{
using std::experimental::range;
}
