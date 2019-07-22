#pragma once
#include <nn/bits/ops/elementary.hpp>
#include <nn/bits/ops/traits.hpp>
#include <nn/common.hpp>

namespace nn::ops
{
template <typename image_order, typename Op> class apply_bias
{
    static constexpr ttl::rank_t r = rank_of<image_order>;
    static constexpr ttl::rank_t p = bias_position<image_order>;

  public:
    shape<r> operator()(const shape<r> &x, const shape<1> &y) const
    {
        contract_assert(std::get<p>(x.dims()) == std::get<0>(y.dims()));
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        Op f;
        const auto s = x.shape();
        for (auto i : range(s.size())) {
            const auto j = s.template coord<p>(i);
            z.data()[i] = f(x.data()[i], y.data()[j]);
        }
    }
};

template <typename image_order>
using add_bias = apply_bias<image_order, scalar_add>;

template <typename image_order>
using mul_bias = apply_bias<image_order, scalar_mul>;

}  // namespace nn::ops
