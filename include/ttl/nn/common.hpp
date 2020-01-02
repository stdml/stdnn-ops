#pragma once
#include <experimental/contract>
#include <experimental/new_type>

#include <ttl/debug>
#include <ttl/device>
#include <ttl/range>
#include <ttl/shape>
#include <ttl/tensor>

namespace std
{
template <ttl::rank_t r>
string to_string(const ttl::shape<r> &s)
{
    return ttl::to_string(s);
}
}  // namespace std

namespace ttl
{
using arity_t = uint8_t;
}  // namespace ttl

template <rank_t r, typename Dim>
void contract_assert_eq_(const internal::basic_shape<r, Dim> &x,
                         const internal::basic_shape<r, Dim> &y,
                         const char *file, int line)
{
    if (x != y) {
        throw std::runtime_error(
            "contract_assert_eq failed: " + std::string(file) + ":" +
            std::to_string(line) + ": " + std::to_string(x) +
            " != " + std::to_string(y));
    }
}
