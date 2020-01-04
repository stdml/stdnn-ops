#pragma once
#include <ttl/nn/bits/ops/loss.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <int>
class xentropy;

template <>
class xentropy<1>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r - 1> &gz, const shape<r - 1> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        return ttl::nn::ops::gradient_shape<1>(ttl::nn::ops::xentropy(), gz, z,
                                               x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gx,
                    const ttl::tensor_view<R, 0> &gz,
                    const ttl::tensor_view<R, 0> &z,
                    const ttl::tensor_view<R, 1> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        for (auto i : range<0>(x)) {
            gx.data()[i] = gz.data()[0] * (-x.data()[i] / y.data()[i]);
        }
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &gx,
                    const ttl::tensor_view<R, 1> &gz,
                    const ttl::tensor_view<R, 1> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 2> &y) const
    {
        for (auto i : range<0>(x)) {
            operator()(gx[i], gz[i], z[i], x[i], y[i]);
        }
    }
};
}  // namespace ttl::nn::ops::grad
