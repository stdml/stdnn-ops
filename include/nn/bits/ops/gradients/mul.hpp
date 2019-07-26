#pragma once
#include <nn/bits/ops/elementary.hpp>
#include <nn/bits/ops/shape_algo.hpp>
#include <nn/common.hpp>

namespace nn::ops::grad
{
template <int> class mul;

template <> class mul<0>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &gz, const shape<r> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        return nn::ops::gradient_shape<0>(nn::ops::mul(), gz, z, x, y);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &gx,
                    const ttl::tensor_view<R, r> &gz,
                    const ttl::tensor_view<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        nn::ops::mul()(gx, gz, y);
    }
};

template <> class mul<1>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &gz, const shape<r> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        return nn::ops::gradient_shape<1>(nn::ops::mul(), gz, z, x, y);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &gy,
                    const ttl::tensor_view<R, r> &gz,
                    const ttl::tensor_view<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        nn::ops::mul()(gy, gz, x);
    }
};

}  // namespace nn::ops::grad
