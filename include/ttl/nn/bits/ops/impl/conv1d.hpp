#pragma once
#include <ttl/nn/bits/ops/conv1d.hpp>

namespace nn::ops
{
template <typename R>
void conv1d::operator()(const ttl::tensor_ref<R, 1> &z,
                        const ttl::tensor_view<R, 1> &x,
                        const ttl::tensor_view<R, 1> &y) const
{
    const filter_t filter(std::get<0>(y.shape().dims()), stride_, rate_);
    const auto [n] = x.shape().dims();

    for (auto j : ttl::range<0>(z)) {
        R xi = 0;
        for (auto k : ttl::range<0>(y)) {
            const auto i = filter(j, k);
            if (padding_.inside(i, n)) {
                xi += y.data()[k] * x.data()[padding_.unpad(i)];
            }
        }
        z.data()[j] = xi;
    }
}
}  // namespace nn::ops
