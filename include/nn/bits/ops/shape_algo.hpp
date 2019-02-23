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
            dims[i] = s.dims[i] + t.dims[i];
        } else {
            contract_assert(s.dims[i] == t.dims[i]);
            dims[i] = s.dims[i];
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
    const dim_t m = std::accumulate(s.dims.begin(), s.dims.begin() + p,
                                    (dim_t)1, std::multiplies<dim_t>());
    const dim_t n = std::accumulate(s.dims.begin() + p, s.dims.end(), (dim_t)1,
                                    std::multiplies<dim_t>());
    return shape<2>(m, n);
}

}  // namespace nn::ops
