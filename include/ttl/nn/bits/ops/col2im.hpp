#pragma once
#include <ttl/algorithm>
#include <ttl/nn/bits/ops/im2col.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/bits/traits/multi_linear_sample.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
template <typename image_order, typename col_order>
class col2im;

template <>
class col2im<hwc, hwrsc> : public im2col_trait<hw>
{
    using im2col_trait::im2col_trait;

  public:
    // shape<3> operator()(const shape<5> &x) const
    // {
    //     const auto [h_, w_, r, s, c] = x.shape().dims();
    //     contract_assert_eq(get_ksize(), shape<2>(r, t));
    // }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 3> &y,
                    const ttl::tensor_view<R, 5> &x) const
    {
        const auto [h_, w_, r, s, c] = x.shape().dims();
        const auto [h, w, _c] = y.shape().dims();
        contract_assert_eq(_c, c);
        const sample_t &h_sample_ = std::get<0>(samples_);
        const sample_t &w_sample_ = std::get<1>(samples_);
        ttl::fill(y, static_cast<R>(0));
        for (const auto i_ : range(h_)) {
            for (const auto j_ : range(w_)) {
                for (const auto u : range(r)) {
                    for (const auto v : range(s)) {
                        for (const auto k : range(c)) {
                            const auto i = h_sample_(i_, u);
                            const auto j = w_sample_(j_, v);
                            if (h_sample_.inside(i, h) &&
                                w_sample_.inside(j, w)) {
                                y.at(h_sample_.unpad(i), w_sample_.unpad(j),
                                     k) += x.at(i_, j_, u, v, k);
                            }
                        }
                    }
                }
            }
        }
    }
};
}  // namespace ttl::nn::ops
