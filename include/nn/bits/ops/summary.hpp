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

struct sum {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return std::accumulate(x.data(), x.data() + x.shape().size(), (R)0,
                               std::plus<R>());
    }
};

struct mean {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        return sum()(x) / x.shape().size();
    }
};

#ifdef HAVE_transform_reduce
struct variance_impl_tr {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        static_assert(std::is_floating_point<R>::value);
        const R m = sum()(x) / x.shape().size();
        const R n_var =
            std::transform_reduce(x.data(), x.data() + x.shape().size(), (R)0,
                                  std::plus<R>(), [m](R xi) {
                                      const R d = xi - m;
                                      return d * d;
                                  });
        return n_var / x.shape().size();
    }
};
#endif

struct variance_impl_old {
    template <typename R, ttl::rank_t r>
    R operator()(const ttl::tensor_view<R, r> &x) const
    {
        static_assert(std::is_floating_point<R>::value);
        const auto n = x.shape().size();
        const R m = sum()(x) / n;
        R n_var = 0;
        for (auto i : range(n)) {
            const R d = x.data()[i] - m;
            n_var += d * d;
        }
        return n_var / n;
    }
};

using variance = variance_impl_old;

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
        R y = 0;
        for (auto i : range(x.shape().size() - 1)) {
            y += std::fabs(x.data()[i + 1] - x.data()[i]);
        }
        return y;
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
