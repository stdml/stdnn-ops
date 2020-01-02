#pragma once
#include <cstring>

#include <algorithm>
#include <type_traits>
#include <vector>

#include <ttl/algorithm>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class cast : public endofunction
{
  public:
    using endofunction::operator();

    template <typename R, typename S, rank_t r>
    void operator()(const tensor_ref<R, r> &y, const tensor_view<S, r> &x) const
    {
        ttl::cast(x, y);
    }
};

class argmax : public reduce_function
{
  public:
    using reduce_function::operator();

    template <typename R, typename N, rank_t r>
    void operator()(const tensor_ref<N, r> &y,
                    const tensor_view<R, r + 1> &x) const
    {
        const auto x_flat = as_matrix<r, 1>(x);
        for (auto i : range(y.shape().size())) {
            y.data()[i] = ttl::argmax(x_flat[i]);
        }
    }
};

class onehot : public vectorize_function
{
  public:
    using vectorize_function::operator();

    onehot(const dim_t k) : vectorize_function(k) {}

    template <typename R, typename N, rank_t r>
    void operator()(const tensor_ref<R, r + 1> &y,
                    const tensor_view<N, r> &x) const
    {
        constexpr R def = 0;
        std::fill(y.data(), y.data_end(), def);
        const auto y_flat = ttl::nn::ops::as_matrix<r, 1>(y);
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
    template <rank_t r>
    shape<0> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert_eq(x, y);
        return shape<0>();
    }

    template <typename R, typename R1, rank_t r>
    void operator()(const tensor_ref<R, 0> &z, const tensor_view<R1, r> &x,
                    const tensor_view<R1, r> &y) const
    {
        static_assert(std::is_floating_point<R>::value);
        z.data()[0] = 1 - static_cast<R>(ttl::hamming_distance(x, y)) /
                              static_cast<R>(x.shape().size());
    }
};

class top
{
    using dim_t = shape<1>::dimension_type;

    const dim_t k_;

  public:
    top(int k) : k_(k) {}

    template <typename R>
    void operator()(const tensor_ref<R, 1> &y, const tensor_ref<dim_t, 1> &z,
                    const tensor_view<R, 1> &x) const
    {
        const dim_t n = x.shape().size();

        using vt = std::vector<std::pair<R, dim_t>>;
        vt v(n);
        for (auto i : range(n)) {
            v[i].first = x.at(i);
            v[i].second = i;
        }
        std::sort(v.begin(), v.end(), std::greater<typename vt::value_type>());

        for (auto i : range(std::min(n, k_))) {
            y.at(i) = v[i].first;
            z.at(i) = v[i].second;
        }
    }
};
}  // namespace ttl::nn::ops
