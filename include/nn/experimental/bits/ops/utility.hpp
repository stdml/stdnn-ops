#pragma once
#include <cstring>

#include <algorithm>

#include <ttl/algorithm>

#include <nn/bits/ops/reshape.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops
{

class cast : public nn::ops::endofunction
{
  public:
    template <typename R, typename S, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<S, r> &x) const
    {
        ttl::cast(x, y);
    }
};

// TODO: make it an operator
// template <typename R, ttl::rank_t r>
// void fill(const ttl::tensor_ref<R, r> &t, R val)
// {
//     ttl::fill(t, val);
// }

class argmax : public nn::ops::reduce_function
{
  public:
    template <typename R, typename N, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<N, r> &y,
                    const ttl::tensor_view<R, r + 1> &x)
    {
        const auto x_flat = nn::ops::as_matrix<r, 1, ttl::tensor_view<R, 2>>(x);
        for (auto i : range(y.shape().size())) {
            y.data()[i] = ttl::argmax(x_flat[i]);
        }
    }
};

class onehot : public nn::ops::vectorize_function
{
  public:
    onehot(const dim_t k) : vectorize_function(k) {}

    template <typename R, typename N, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r + 1> &y,
                    const ttl::tensor_view<N, r> &x) const
    {
        constexpr R def = 0;
        std::fill(y.data(), y.data_end(), def);
        const auto y_flat = nn::ops::as_matrix<r, 1, ttl::tensor_ref<R, 2>>(y);
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
}  // namespace nn::experimental::ops
