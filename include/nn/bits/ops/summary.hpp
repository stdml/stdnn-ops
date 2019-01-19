#pragma once
#include <algorithm>
#include <cmath>

#include <nn/common.hpp>

namespace nn::ops
{
namespace summaries
{
struct min {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return *std::min_element(x.data(), x.data() + x.shape().size());
    }
};

struct max {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return *std::max_element(x.data(), x.data() + x.shape().size());
    }
};

struct span {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        const auto mm =
            std::minmax_element(x.data(), x.data() + x.shape().size());
        return *mm.second - *mm.first;
    }
};

struct sum {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return std::accumulate(x.data(), x.data() + x.shape().size(),
                               static_cast<R>(0));
    }
};

struct mean {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return sum()(x) / x.shape().size();
    }
};

struct variance {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        static_assert(std::is_floating_point<R>::value);
        const R n_var = std::accumulate(
            x.data(), x.data() + x.shape().size(), static_cast<R>(0),
            [m = sum()(x) / x.shape().size()](R acc, R xi) {
                const R d = xi - m;
                return acc + d * d;
            });
        return n_var / x.shape().size();
    }
};

struct standard_deviation {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        static_assert(std::is_floating_point<R>::value);
        return std::sqrt(variance()(x));
    }
};

struct adj_diff_sum {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return std::inner_product(x.data() + 1, x.data() + x.shape().size(),
                                  x.data(), static_cast<R>(0), std::plus<R>(),
                                  [](R x2, R x1) { return std::abs(x2 - x1); });
    }
};

using var = variance;
using std = standard_deviation;

}  // namespace summaries

template <typename... S> class scalar_summaries
{
    static constexpr int n_indexes = sizeof...(S);

  public:
    template <ttl::rank_t r> shape<1> operator()(const shape<r> &x) const
    {
        return shape<1>(n_indexes);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, 1> &y,
                    const ttl::tensor_view<R, r> &x) const
    {
        std::array<R, n_indexes> values({(S()(x))...});
        std::copy(values.begin(), values.end(), y.data());
    }
};
}  // namespace nn::ops
