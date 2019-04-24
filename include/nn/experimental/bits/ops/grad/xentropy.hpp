#pragma once
#include <nn/common.hpp>

#include <experimental/range>

using std::experimental::range;

namespace nn::experimental::ops::grad
{
template <int> class xentropy;

template <> class xentropy<1>
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r - 1> &gz, const shape<r - 1> &z,
                        const shape<r> &x, const shape<r> &y) const
    {
        // TODO: check
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gx,
                    const ttl::tensor_view<R, 0> &gz,
                    const ttl::tensor_view<R, 0> &z,
                    const ttl::tensor_view<R, 1> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        for (auto i : range(y.shape().size())) {
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
        for (auto i : range(y.shape().dims[0])) {
            operator()(gx[i], gz[i], z[i], x[i], y[i]);
        }
    }
};
}  // namespace nn::experimental::ops::grad
