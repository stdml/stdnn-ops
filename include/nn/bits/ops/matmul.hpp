#pragma once
#include <algorithm>
#include <experimental/contract>
#include <stdtensor>

#include <nn/bits/engines/linag.hpp>

namespace nn::ops
{
class matmul
{
  public:
    shape<2> operator()(const shape<2> &x, const shape<2> &y) const
    {
        const auto [n, m] = x.dims;
        const auto [_m, k] = y.dims;
        contract_assert(m == _m);
        return shape<2>(n, k);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 2> &y) const
    {
        nn::engines::linag<R>::mm(x, y, z);
    }
};

}  // namespace nn::ops
