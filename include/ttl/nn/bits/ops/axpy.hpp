#pragma once
#include <ttl/nn/common.hpp>

namespace nn::ops
{

class axpy
{
  public:
    template <ttl::rank_t r>
    ttl::shape<r> operator()(const ttl::shape<0> &a, const ttl::shape<r> &x,
                             const ttl::shape<r> &y) const
    {
        contract_assert_eq(x, y);
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &z,
                    const ttl::tensor_view<R, 0> &a,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        std::transform(
            x.data(), x.data_end(), y.data(), z.data(),
            [a = a.data()[0]](const R &x, const R &y) { return a * x + y; });
    }
};
}  // namespace nn::ops
