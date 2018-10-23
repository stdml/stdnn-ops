#pragma once
#include <algorithm>
#include <experimental/contract>
#include <stdtensor>

namespace nn::ops
{
class add
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert(x == y);
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        const auto n = x.shape().size();
        std::transform(x.data(), x.data() + n, y.data(), z.data(),
                       std::plus<R>());
    }
};

}  // namespace nn::ops
