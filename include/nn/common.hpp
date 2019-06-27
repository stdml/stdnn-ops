#pragma once
#include <experimental/contract>
#include <experimental/new_type>
#include <experimental/range>

#include <ttl/debug>
#include <ttl/tensor>

namespace std
{
template <ttl::rank_t r> string to_string(const ttl::shape<r> &s)
{
    return ttl::to_string(s);
}
}  // namespace std

namespace nn
{
using ttl::shape;

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
