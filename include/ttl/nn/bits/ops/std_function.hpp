#pragma once
#include <ttl/bits/std_shape.hpp>

namespace ttl
{
using internal::basic_shape;
using internal::rank_t;

namespace nn::ops
{
namespace internal
{
class basic_endofunction
{
  public:
    template <rank_t r, typename Dim>
    basic_shape<r, Dim> operator()(const basic_shape<r, Dim> &x) const
    {
        return x;
    }
};

class basic_binary_endofunction
{
  public:
    template <rank_t r, typename Dim>
    basic_shape<r, Dim> operator()(const basic_shape<r, Dim> &x,
                                   const basic_shape<r, Dim> &y) const
    {
        if (x != y) { throw std::invalid_argument("shapes don't match"); }
        return x;
    }
};

class basic_reduce_function
{
  public:
    template <rank_t r, typename Dim>
    basic_shape<r - 1, Dim> operator()(const basic_shape<r, Dim> &s) const
    {
        std::array<Dim, r - 1> dims;
        std::copy(s.dims().begin(), s.dims().end() - 1, dims.begin());
        return basic_shape<r - 1, Dim>(dims);
    }
};

class basic_binary_reduce_function
{
  public:
    template <rank_t r, typename Dim>
    basic_shape<r - 1, Dim> operator()(const basic_shape<r, Dim> &s,
                                       const basic_shape<r, Dim> &t) const
    {
        if (s != t) { throw std::invalid_argument("shapes don't match"); }
        std::array<Dim, r - 1> dims;
        std::copy(s.dims().begin(), s.dims().end() - 1, dims.begin());
        return basic_shape<r - 1, Dim>(dims);
    }
};

class basic_vectorize_function
{
  protected:
    using dim_t = basic_shape<0>::dimension_type;

    const dim_t k_;

  public:
    basic_vectorize_function(const dim_t k) : k_(k) {}

    template <rank_t r, typename Dim>
    basic_shape<r + 1, Dim> operator()(const basic_shape<r, Dim> &s) const
    {
        std::array<Dim, r + 1> dims;
        std::copy(s.dims().begin(), s.dims().end(), dims.begin());
        dims[r] = k_;
        return basic_shape<r + 1, Dim>(dims);
    }
};
}  // namespace internal

using endofunction = internal::basic_endofunction;
using binary_endofunction = internal::basic_binary_endofunction;
using reduce_function = internal::basic_reduce_function;
using binary_reduce_function = internal::basic_binary_reduce_function;
using vectorize_function = internal::basic_vectorize_function;
}  // namespace nn::ops
}  // namespace ttl
