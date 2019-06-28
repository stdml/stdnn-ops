#pragma once
#include <algorithm>

#include <nn/bits/ops/elementary.hpp>
#include <nn/bits/ops/shape_algo.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops::grad
{
template <int> class add;

template <> class add<0>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &gz, const shape<r> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        return nn::ops::gradient_shape<0>(nn::ops::add(), gz, z, x, y);
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
        return nn::ops::gradient_shape<1>(nn::ops::add(), gz, z, x, y);
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

}  // namespace nn::experimental::ops::grad
