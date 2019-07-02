#pragma once
#include <ttl/algorithm>

#include <nn/bits/ops/bias.hpp>
#include <nn/bits/ops/reduce.hpp>
#include <nn/bits/ops/shape_algo.hpp>
#include <nn/bits/ops/traits.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops::grad
{

template <typename image_order, int> class add_bias;

template <> class add_bias<nn::ops::hw, 0>
{
  public:
    shape<2> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<1> &y) const
    {
        return nn::ops::gradient_shape<0>(nn::ops::add_bias<nn::ops::hw>(), gz,
                                          z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &gx,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        nn::ops::identity()(gx, gz);
    }
};

template <> class add_bias<nn::ops::hw, 1>
{
  public:
    shape<1> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<1> &y) const
    {
        return nn::ops::gradient_shape<1>(nn::ops::add_bias<nn::ops::hw>(), gz,
                                          z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gy,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        nn::ops::internal::outter_contraction(gy, gz);
    }
};

template <> class add_bias<nn::ops::nhwc, 0>
{
  public:
    shape<4> operator()(const shape<4> &gz, const shape<4> &z,
                        const shape<4> &x, const shape<1> &y) const
    {
        return nn::ops::gradient_shape<0>(nn::ops::add_bias<nn::ops::nhwc>(),
                                          gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &gx,
                    const ttl::tensor_view<R, 4> &gz,
                    const ttl::tensor_view<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        nn::ops::identity()(gx, gz);
    }
};

template <> class add_bias<nn::ops::nhwc, 1>
{
  public:
    shape<1> operator()(const shape<4> &gz, const shape<4> &z,
                        const shape<4> &x, const shape<1> &y) const
    {
        return nn::ops::gradient_shape<1>(nn::ops::add_bias<nn::ops::nhwc>(),
                                          gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gy,
                    const ttl::tensor_view<R, 4> &gz,
                    const ttl::tensor_view<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        nn::ops::internal::outter_contraction(gy, nn::ops::as_matrix<3, 1>(gz));
    }
};

}  // namespace nn::experimental::ops::grad
