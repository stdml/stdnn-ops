#pragma once
#include <array>
#include <nn/common.hpp>

namespace nn::ops
{
namespace internal
{

template <typename T, size_t p, size_t q, size_t... Is, size_t... Js>
constexpr std::array<T, p + q>
merge_indexed(const std::array<T, p> &a, std::index_sequence<Is...>,
              const std::array<T, q> &b, std::index_sequence<Js...>)
{
    return std::array<T, p + q>{std::get<Is>(a)..., std::get<Js>(b)...};
}

template <ttl::rank_t p, ttl::rank_t q>
constexpr shape<p + q> join_shape(const shape<p> &s, const shape<q> &t)
{
    return shape<p + q>(merge_indexed(s.dims, std::make_index_sequence<p>(),
                                      t.dims, std::make_index_sequence<q>()));
}

template <ttl::rank_t r>
shape<r + 1> batch(typename shape<r>::dimension_type n, const shape<r> &s)
{
    return internal::join_shape(shape<1>(n), s);
}

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
