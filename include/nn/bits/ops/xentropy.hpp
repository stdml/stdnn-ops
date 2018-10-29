#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>

#include <experimental/contract>
#include <stdtensor>

namespace nn::ops
{
namespace internal
{
template <typename R> struct xentropy {
    const size_t size;

    xentropy(size_t size) : size(size) {}

    R operator()(const R *x, const R *y) const
    {
        return -std::inner_product(x, x + size, y, static_cast<R>(0),
                                   std::plus<R>(),
                                   [](R x, R y) { return x * std::log(y); });
    }
};
}  // namespace internal

template <ttl::rank_t r> class xentropy;

template <> class xentropy<1>
{
  public:
    shape<0> operator()(const shape<1> &x, const shape<1> &y) const
    {
        contract_assert(x == y);
        return x.template subshape<1>();
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r - 1> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        static_assert(r == 1);
        const auto k = x.shape().size();
        z.data()[0] = (internal::xentropy<R>(k))(x.data(), y.data());
    }
};

template <ttl::rank_t r> class xentropy
{
  public:
    shape<r - 1> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert(x == y);
        return x.template subshape<1>();
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, r - 1> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        static_assert(r == 2);  // TODO: support higher ranks
        const auto [n, k] = x.shape().dims;
        for (auto i : range(n)) {
            z.at(i) = (internal::xentropy<R>(k))(x[i].data(), y[i].data());
        }
    }
};

}  // namespace nn::ops
