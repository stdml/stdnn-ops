#pragma once
#include <nn/bits/engines/linag.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops::grad
{

template <int, typename E = nn::engines::default_engine> class matmul;

template <typename E> class matmul<0, E>
{
  public:
    shape<2> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<2> &y) const
    {
        // TODO: check
        return x;
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
        // TODO: check
        return y;
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
}  // namespace nn::experimental::ops::grad
