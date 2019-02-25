#pragma once
#include <array>

#include <nn/bits/ops/shape_algo.hpp>

namespace nn::ops
{

template <ttl::rank_t p, ttl::rank_t q, typename T, typename T1>
T as_matrix(const T1 &t)
{
    static_assert(T::rank == 2);
    static_assert(T1::rank == p + q);
    return T(t.data(), as_mat_shape<p, q>(t.shape()));
}

// TODO: make it generic: template <ttl::rank_t ... ranks>
template <ttl::rank_t p, ttl::rank_t q> class flatten
{
    static constexpr ttl::rank_t r = p + q;

  public:
    shape<2> operator()(const shape<r> &x) const
    {
        return as_mat_shape<p, q>(x);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, 2> &y,
                    const ttl::tensor_view<R, r> &x) const
    {
        std::copy(x.data(), x.data() + x.shape().size(), y.data());
    }
};

}  // namespace nn::ops
