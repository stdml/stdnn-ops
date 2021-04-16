#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/conv.hpp>
#include <ttl/nn/bits/traits/conv_traits.hpp>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename image_order, typename col_order>
class im2col_idx_map;

template <>
class im2col_idx_map<traits::hwc, traits::hwrsc>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    im2col_idx_map(const im2col_trait &trait) : im2col_trait(trait) {}

    template <typename N, typename Dim>
    void operator()(const tensor_ref<N, 5> &y,
                    const ttl::internal::basic_shape<3, Dim> &x) const
    {
        static_assert(std::is_signed<N>::value, "");
        const auto [h, w, _c] = x.dims();
        static_assert(sizeof(_c) > 0, "");  // unused
        // const auto [h_, w_, r, s, _c] = y.dims();
        const auto &[h_sample, w_sample] = samples_;
        const auto shape = y.shape();
        for (auto idx : range(y.size())) {
            const auto [i_, j_, u, v, k] = shape.expand(idx);
            const auto i = h_sample(i_, u);
            const auto j = w_sample(j_, v);
            if (h_sample.inside(i, h) && w_sample.inside(j, w)) {
                y.data()[idx] =
                    x.offset(h_sample.unpad(i), w_sample.unpad(j), k);
            } else {
                y.data()[idx] = -1;
            }
        }
    }

    template <typename R, typename N>
    void operator()(const tensor_ref<R, 5> &z, const tensor_view<N, 5> &y,
                    const tensor_view<R, 3> &x) const
    {
        for (auto i : range(z.size())) {
            if (const auto j = y.data()[i]; j >= 0) {
                z.data()[i] = x.data()[j];
            } else {
                z.data()[i] = 0;  // ! important fix
            }
        }
    }
};

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

template <typename R>
class im2col_2d<host_memory, traits::nhwc, traits::nhwrsc, R>
    : public traits::im2col_trait<traits::hw>
{
    using im2col_trait::im2col_trait;

  public:
    im2col_2d(const im2col_trait &trait) : im2col_trait(trait) {}

    void operator()(const tensor_ref<R, 6> &y, const tensor_view<R, 4> &x) const
    {
        const im2col_trait &trait = *this;
        im2col_idx_map<traits::hwc, traits::hwrsc> idx_map(trait);
        tensor<int, 5> q(y.shape().subshape());
        idx_map(ref(q), x.shape().subshape());
        for (auto i : range<0>(x)) { idx_map(y[i], view(q), x[i]); }
    }
};
}  // namespace ttl::nn::kernels
