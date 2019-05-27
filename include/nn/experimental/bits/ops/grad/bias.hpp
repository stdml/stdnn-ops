#pragma once
#include <nn/bits/engines/linag.hpp>
#include <nn/bits/ops/bias.hpp>
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
        contract_assert_eq(z, nn::ops::add_bias<nn::ops::hw>()(x, y));
        contract_assert_eq(z, gz);
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &gx,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        std::copy(gz.data(), gz.data() + gz.shape().size(), gx.data());
    }
};

template <> class add_bias<nn::ops::hw, 1>
{
  public:
    shape<1> operator()(const shape<2> &gz, const shape<2> &z,
                        const shape<2> &x, const shape<1> &y) const
    {
        contract_assert_eq(z, nn::ops::add_bias<nn::ops::hw>()(x, y));
        contract_assert_eq(z, gz);
        return y;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gy,
                    const ttl::tensor_view<R, 2> &gz,
                    const ttl::tensor_view<R, 2> &z,
                    const ttl::tensor_view<R, 2> &x,
                    const ttl::tensor_view<R, 1> &y) const
    {
        std::fill(gy.data(), gy.data() + gy.shape().size(), 0);
        for (const auto gzi : gz) {
            std::transform(gy.data(), gy.data() + gy.shape().size(), gzi.data(),
                           gy.data(), std::plus<R>());
        }
    }
};

}  // namespace nn::experimental::ops::grad
