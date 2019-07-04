#pragma once
#include <array>

#include <nn/common.hpp>

namespace nn::ops
{
namespace internal
{

template <ttl::rank_t p, ttl::rank_t r>
shape<r> concat2shapes(const shape<r> &s, const shape<r> &t)
{
    using dim_t = typename shape<r>::dimension_type;
    std::array<dim_t, r> dims;
    for (auto i : range(r)) {
        if (i == p) {
            dims[i] = s.dims()[i] + t.dims()[i];
        } else {
            contract_assert(s.dims()[i] == t.dims()[i]);
            dims[i] = s.dims()[i];
        }
    }
    return shape<r>(dims);
}

template <ttl::rank_t p, ttl::rank_t r> shape<r> concat_shape(const shape<r> &s)
{
    return s;
}

template <ttl::rank_t p, ttl::rank_t r, typename... S>
shape<r> concat_shape(const shape<r> &s, const S &... ss)
{
    return concat2shapes<p, r>(s, concat_shape<p, r>(ss...));
}

}  // namespace internal

template <ttl::rank_t p, ttl::rank_t q>
shape<2> as_mat_shape(const shape<p + q> &s)
{
    using dim_t = typename shape<p + q>::dimension_type;
    const dim_t m = std::accumulate(s.dims().begin(), s.dims().begin() + p,
                                    (dim_t)1, std::multiplies<dim_t>());
    const dim_t n = std::accumulate(s.dims().begin() + p, s.dims().end(),
                                    (dim_t)1, std::multiplies<dim_t>());
    return shape<2>(m, n);
}

class endofunction
{
  public:
    template <ttl::rank_t r> shape<r> operator()(const shape<r> &x) const
    {
        return x;
    }
};

class reduce_function
{
  public:
    template <ttl::rank_t r> shape<r - 1> operator()(const shape<r> &s) const
    {
        std::array<typename shape<r - 1>::dimension_type, r - 1> dims;
        std::copy(s.dims().begin(), s.dims().end() - 1, dims.begin());
        return shape<r - 1>(dims);
    }
};

class vectorize_function
{
  protected:
    using dim_t = shape<0>::dimension_type;
    const dim_t k_;

  public:
    vectorize_function(const dim_t k) : k_(k) {}

    template <ttl::rank_t r> shape<r + 1> operator()(const shape<r> &s) const
    {
        std::array<dim_t, r + 1> dims;
        std::copy(s.dims().begin(), s.dims().end(), dims.begin());
        dims[r] = k_;
        return shape<r + 1>(dims);
    }
};

template <typename F, typename Y, typename... Xs>
void check_shape(const F &f, const Y &y, const Xs &... xs)
{
    if (y.shape() != f(xs.shape()...)) {
        throw std::logic_error("shape check failed");
    }
}

template <arity_t i, typename Op, ttl::rank_t ry, ttl::rank_t... rx>
auto gradient_shape(const Op &infer, const shape<ry> &gy, const shape<ry> &y,
                    const shape<rx> &... xs)
{
    contract_assert_eq(y, infer(xs...));
    contract_assert_eq(y, gy);
    return std::get<i>(std::make_tuple(xs...));
};

}  // namespace nn::ops
