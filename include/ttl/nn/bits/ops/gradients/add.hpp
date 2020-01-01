#pragma once
#include <algorithm>

#include <ttl/nn/bits/ops/elementary.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops::grad
{
template <int> class add;

template <> class add<0>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &gz, const shape<r> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        return ttl::nn::ops::gradient_shape<0>(ttl::nn::ops::add(), gz, z, x,
                                               y);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &gx,
                    const ttl::tensor_view<R, r> &gz,
                    const ttl::tensor_view<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        std::copy(gz.data(), gz.data() + gz.shape().size(), gx.data());
    }
};

template <> class add<1>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &gz, const shape<r> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        return ttl::nn::ops::gradient_shape<1>(ttl::nn::ops::add(), gz, z, x,
                                               y);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &gy,
                    const ttl::tensor_view<R, r> &gz,
                    const ttl::tensor_view<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        std::copy(gz.data(), gz.data() + gz.shape().size(), gy.data());
    }
};

}  // namespace ttl::nn::ops::grad
