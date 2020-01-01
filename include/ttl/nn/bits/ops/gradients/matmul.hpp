#pragma once
#include <ttl/nn/bits/engines/linag.hpp>
#include <ttl/nn/bits/ops/matmul.hpp>
#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>

namespace nn::ops::grad
{

template <int, typename E = nn::engines::default_engine> class matmul;

template <typename E> class matmul<0, E>
{
  public:
    shape<2> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<2> &y) const
    {
        return nn::ops::gradient_shape<0>(nn::ops::matmul(), gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &gx,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 2> &y) const
    {
        nn::engines::linag<E>::mmt(gz, y, gx);
    }
};

template <typename E> class matmul<1, E>
{
  public:
    shape<2> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<2> &y) const
    {
        return nn::ops::gradient_shape<1>(nn::ops::matmul(), gz, z, x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &gy,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 2> &y) const
    {
        nn::engines::linag<E>::mtm(x, gz, gy);
    }
};
}  // namespace nn::ops::grad
