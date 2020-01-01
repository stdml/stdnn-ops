#pragma once
#include <cmath>

#include <algorithm>
#include <experimental/contract>
#include <numeric>

#include <ttl/nn/bits/ops/reshape.hpp>

namespace ttl::nn::ops
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

class xentropy
{
  public:
    template <ttl::rank_t r>
    shape<r - 1> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert(x == y);
        std::array<typename shape<r - 1>::dimension_type, r - 1> dims;
        std::copy(x.dims().begin(), x.dims().end() - 1, dims.begin());
        return shape<r - 1>(dims);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r - 1> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        const auto x_flat = as_matrix<r - 1, 1>(x);
        const auto y_flat = as_matrix<r - 1, 1>(y);
        const auto [n, k] = x_flat.shape().dims();
        for (auto i : range(n)) {
            z.data()[i] =
                (internal::xentropy<R>(k))(x_flat[i].data(), y_flat[i].data());
        }
    }
};

}  // namespace ttl::nn::ops
