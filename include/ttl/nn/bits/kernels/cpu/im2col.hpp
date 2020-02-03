#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/conv.hpp>
#include <ttl/nn/bits/traits/conv_traits.hpp>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class im2col_2d<host_memory, traits::hw, traits::hwrs, R>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    im2col_2d(const im2col_trait &trait) : im2col_trait(trait) {}

    void operator()(const tensor_ref<R, 4> &y, const tensor_view<R, 2> &x) const
    {
        const auto [h, w] = x.dims();
        const auto [h_, w_, r, s] = y.dims();
        const auto &[h_sample, w_sample] = samples_;
        for (const auto i_ : range(h_)) {
            for (const auto j_ : range(w_)) {
                for (const auto u : range(r)) {
                    for (const auto v : range(s)) {
                        R value = 0;
                        const auto i = h_sample(i_, u);
                        const auto j = w_sample(j_, v);
                        if (h_sample.inside(i, h) && w_sample.inside(j, w)) {
                            value = x.at(h_sample.unpad(i), w_sample.unpad(j));
                        }
                        y.at(i_, j_, u, v) = value;
                    }
                }
            }
        }
    }
};

template <typename R>
class im2col_2d<host_memory, traits::hw, traits::rshw, R>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    im2col_2d(const im2col_trait &trait) : im2col_trait(trait) {}

    void operator()(const tensor_ref<R, 4> &y, const tensor_view<R, 2> &x) const
    {
        const auto [h, w] = x.dims();
        const auto [r, s, h_, w_] = y.dims();
        const auto &[h_sample, w_sample] = samples_;
        for (const auto u : range(r)) {
            for (const auto v : range(s)) {
                for (const auto i_ : range(h_)) {
                    for (const auto j_ : range(w_)) {
                        R value = 0;
                        const auto i = h_sample(i_, u);
                        const auto j = w_sample(j_, v);
                        if (h_sample.inside(i, h) && w_sample.inside(j, w)) {
                            value = x.at(h_sample.unpad(i), w_sample.unpad(j));
                        }
                        y.at(u, v, i_, j_) = value;
                    }
                }
            }
        }
    }
};

template <typename R>
class im2col_2d<host_memory, traits::hwc, traits::hwrsc, R>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    im2col_2d(const im2col_trait &trait) : im2col_trait(trait) {}

    void operator()(const tensor_ref<R, 5> &y, const tensor_view<R, 3> &x) const
    {
        const auto [h, w, c] = x.dims();
        const auto [h_, w_, r, s, _c] = y.dims();
        contract_assert(_c == c);
        const auto &[h_sample, w_sample] = samples_;
        for (const auto i_ : range(h_)) {
            for (const auto j_ : range(w_)) {
                for (const auto u : range(r)) {
                    for (const auto v : range(s)) {
                        for (const auto k : range(c)) {
                            R value = 0;
                            const auto i = h_sample(i_, u);
                            const auto j = w_sample(j_, v);
                            if (h_sample.inside(i, h) &&
                                w_sample.inside(j, w)) {
                                value = x.at(h_sample.unpad(i),
                                             w_sample.unpad(j), k);
                            }
                            y.at(i_, j_, u, v, k) = value;
                        }
                    }
                }
            }
        }
    }
};
}  // namespace ttl::nn::kernels
