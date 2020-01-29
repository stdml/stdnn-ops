#pragma once
#include <ttl/algorithm>
#include <ttl/device>
#include <ttl/nn/bits/kernels/conv.hpp>
#include <ttl/nn/bits/traits/conv_traits.hpp>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class col2im_2d<host_memory, traits::hwc, traits::hwrsc, R>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    col2im_2d(const im2col_trait &trait) : im2col_trait(trait) {}

    void operator()(const tensor_ref<R, 3> &y, const tensor_view<R, 5> &x) const
    {
        const auto [h_, w_, r, s, c] = x.dims();
        const auto [h, w, _c] = y.dims();
        contract_assert_eq(_c, c);
        const auto &[h_sample, w_sample] = samples_;
        ttl::fill(y, static_cast<R>(0));
        for (const auto i_ : range(h_)) {
            for (const auto j_ : range(w_)) {
                for (const auto u : range(r)) {
                    for (const auto v : range(s)) {
                        for (const auto k : range(c)) {
                            const auto i = h_sample(i_, u);
                            const auto j = w_sample(j_, v);
                            if (h_sample.inside(i, h) &&
                                w_sample.inside(j, w)) {
                                y.at(h_sample.unpad(i), w_sample.unpad(j), k) +=
                                    x.at(i_, j_, u, v, k);
                            }
                        }
                    }
                }
            }
        }
    }
};
}  // namespace ttl::nn::kernels
