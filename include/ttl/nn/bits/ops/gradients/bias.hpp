#pragma once
#include <ttl/algorithm>
#include <ttl/nn/bits/ops/bias.hpp>
#include <ttl/nn/bits/ops/reduce.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops::grad
{

template <typename image_order, int> class add_bias;

template <typename image_order> class add_bias<image_order, 0>
{
    static constexpr auto r = ttl::nn::ops::rank_of<image_order>;
    static constexpr auto p = ttl::nn::ops::bias_position<image_order>;

  public:
    shape<2> operator()(const shape<r> &gz, const shape<r> &z,
                        const shape<r> &x, const shape<1> &y) const
    {
        return ttl::nn::ops::gradient_shape<0>(
            ttl::nn::ops::add_bias<image_order>(), gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, r> &gx,
                    const ttl::tensor_view<R, r> &gz,
                    const ttl::tensor_view<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        ttl::nn::ops::identity()(gx, gz);
    }
};

template <> class add_bias<ttl::nn::ops::hw, 1>
{
  public:
    shape<1> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<1> &y) const
    {
        return ttl::nn::ops::gradient_shape<1>(
            ttl::nn::ops::add_bias<ttl::nn::ops::hw>(), gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gy,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        ttl::nn::ops::internal::outter_contraction(gy, gz);
    }
};

template <> class add_bias<ttl::nn::ops::nhwc, 1>
{
  public:
    shape<1> operator()(const shape<4> &gz, const shape<4> &z,
                        const shape<4> &x, const shape<1> &y) const
    {
        return ttl::nn::ops::gradient_shape<1>(
            ttl::nn::ops::add_bias<ttl::nn::ops::nhwc>(), gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gy,
                    const ttl::tensor_view<R, 4> &gz,
                    const ttl::tensor_view<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        ttl::nn::ops::internal::outter_contraction(
            gy, ttl::nn::ops::as_matrix<3, 1>(gz));
    }
};

}  // namespace ttl::nn::ops::grad
