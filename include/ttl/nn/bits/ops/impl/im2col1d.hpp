#pragma once
#include <ttl/nn/bits/ops/im2col1d.hpp>

namespace ttl::nn::ops
{
template <typename R>
void im2col1d::operator()(const ttl::tensor_ref<R, 2> &y,
                          const ttl::tensor_view<R, 1> &x) const
{
    const R def = 0;
    const auto [n] = x.shape().dims();
    for (auto j : ttl::range<0>(y)) {
        for (auto k : ttl::range<1>(y)) {
            const int i = (*this)(j, k);
            if (inside(i, n)) {
                y.at(j, k) = x.at(unpad(i));
            } else {
                y.at(j, k) = def;
            }
        }
    }
}
}  // namespace ttl::nn::ops
