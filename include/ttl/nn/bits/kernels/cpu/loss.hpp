#pragma once
#include <cmath>

#include <algorithm>
#include <numeric>

#include <ttl/device>
#include <ttl/nn/bits/kernels/loss.hpp>

namespace ttl::nn::kernels
{
template <typename R>
class xentropy<host_memory, R>
{
  public:
    void operator()(const tensor_ref<R, 0> &z, const tensor_view<R, 1> &x,
                    const tensor_view<R, 1> &y) const
    {
        z.data()[0] = -std::inner_product(
            x.data(), x.data_end(), y.data(), static_cast<R>(0), std::plus<R>(),
            [](R x, R y) { return x * std::log(y); });
    }

    void operator()(const tensor_ref<R, 1> &z, const tensor_view<R, 2> &x,
                    const tensor_view<R, 2> &y) const
    {
        for (auto i : range<0>(z)) { operator()(z[i], x[i], y[i]); }
    }
};
}  // namespace ttl::nn::kernels
