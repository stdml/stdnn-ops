#pragma once
#include <cstring>

#include <algorithm>
#include <type_traits>

#include <ttl/algorithm>

#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/common.hpp>

namespace nn::ops
{

class cast : public nn::ops::endofunction
{
  public:
    using endofunction::operator();

    template <typename R, typename S, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<S, r> &x) const
    {
        ttl::cast(x, y);
    }
};

class argmax : public nn::ops::reduce_function
{
  public:
    using reduce_function::operator();

    template <typename R, typename N, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<N, r> &y,
                    const ttl::tensor_view<R, r + 1> &x) const
    {
        const auto x_flat = nn::ops::as_matrix<r, 1>(x);
        for (auto i : range(y.shape().size())) {
            y.data()[i] = ttl::argmax(x_flat[i]);
        }
    }
};

class onehot : public nn::ops::vectorize_function
{
  public:
    using vectorize_function::operator();

    onehot(const dim_t k) : vectorize_function(k) {}

    template <typename R, typename N, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r + 1> &y,
                    const ttl::tensor_view<N, r> &x) const
    {
        constexpr R def = 0;
        std::fill(y.data(), y.data_end(), def);
        const auto y_flat = nn::ops::as_matrix<r, 1>(y);
        for (auto i : range(x.shape().size())) {
            const dim_t j = x.data()[i];
            if (0 <= j && j < k_) {
                y_flat.at(i, j) = static_cast<R>(1);
            } else {
                // TODO: throw?
            }
        }
    }
};

class similarity
{
  public:
    template <ttl::rank_t r>
    shape<0> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert_eq(x, y);
        return shape<0>();
    }

    template <typename R, typename R1, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, 0> &z,
                    const ttl::tensor_view<R1, r> &x,
                    const ttl::tensor_view<R1, r> &y) const
    {
        static_assert(std::is_floating_point<R>::value);
        z.data()[0] = 1 - static_cast<R>(ttl::hamming_distance(x, y)) /
                              static_cast<R>(x.shape().size());
    }
};
}  // namespace nn::ops
