#pragma once
#include <nn/bits/ops/linear_sample.hpp>
#include <nn/bits/ops/reshape.hpp>
#include <nn/bits/ops/traits.hpp>
#include <nn/common.hpp>

namespace nn::ops
{
template <typename image_order> class im2col_trait;

template <> class im2col_trait<hw> : public multi_linear_sample_trait<2, size_t>
{
    using multi_linear_sample_trait::multi_linear_sample_trait;
};

template <typename image_order, typename col_order> class im2col;

template <> class im2col<hw, hwrs> : public im2col_trait<hw>
{
    using im2col_trait::im2col_trait;

  public:
    shape<4> operator()(const shape<2> &x) const
    {
        return ttl::internal::join_shape(im2col_trait::operator()(x),
                                         get_ksize());
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 2> &x) const
    {
        const auto [h, w] = x.shape().dims();
        const auto [h_, w_, r, s] = y.shape().dims();

        const sample_t &h_sample_ = std::get<0>(samples_);
        const sample_t &w_sample_ = std::get<1>(samples_);

        for (const auto i_ : range(h_)) {
            for (const auto j_ : range(w_)) {
                for (const auto u : range(r)) {
                    for (const auto v : range(s)) {
                        R value = 0;
                        const auto i = h_sample_(i_, u);
                        const auto j = w_sample_(j_, v);
                        if (h_sample_.inside(i, h) && w_sample_.inside(j, w)) {
                            value =
                                x.at(h_sample_.unpad(i), w_sample_.unpad(j));
                        }
                        y.at(i_, j_, u, v) = value;
                    }
                }
            }
        }
    }
};

template <> class im2col<hw, rshw> : public im2col_trait<hw>
{
    using im2col_trait::im2col_trait;

  public:
    shape<4> operator()(const shape<2> &x) const
    {
        return ttl::internal::join_shape(get_ksize(),
                                         im2col_trait::operator()(x));
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 2> &x) const
    {
        const auto [h, w] = x.shape().dims();
        const auto [r, s, h_, w_] = y.shape().dims();

        const sample_t &h_sample_ = std::get<0>(samples_);
        const sample_t &w_sample_ = std::get<1>(samples_);

        for (const auto u : range(r)) {
            for (const auto v : range(s)) {
                for (const auto i_ : range(h_)) {
                    for (const auto j_ : range(w_)) {
                        R value = 0;
                        const auto i = h_sample_(i_, u);
                        const auto j = w_sample_(j_, v);
                        if (h_sample_.inside(i, h) && w_sample_.inside(j, w)) {
                            value =
                                x.at(h_sample_.unpad(i), w_sample_.unpad(j));
                        }
                        y.at(u, v, i_, j_) = value;
                    }
                }
            }
        }
    }
};

// TODO: use vectorize
template <> class im2col<hwc, hwrsc> : public im2col_trait<hw>
{
    using im2col_trait::im2col_trait;

  public:
    shape<5> operator()(const shape<3> &x) const
    {
        const auto [r, s] = get_ksize().dims();
        const auto [h, w, c] = x.dims();
        const auto [h_, w_] = im2col_trait::operator()(shape<2>(h, w)).dims();
        return shape<5>(h_, w_, r, s, c);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 5> &y,
                    const ttl::tensor_view<R, 3> &x) const
    {
        const auto [h, w, c] = x.shape().dims();
        const auto [h_, w_, r, s, _c] = y.shape().dims();
        contract_assert(_c == c);

        const sample_t &h_sample_ = std::get<0>(samples_);
        const sample_t &w_sample_ = std::get<1>(samples_);

        for (const auto i_ : range(h_)) {
            for (const auto j_ : range(w_)) {
                for (const auto u : range(r)) {
                    for (const auto v : range(s)) {
                        for (const auto k : range(c)) {
                            R value = 0;
                            const auto i = h_sample_(i_, u);
                            const auto j = w_sample_(j_, v);
                            if (h_sample_.inside(i, h) &&
                                w_sample_.inside(j, w)) {
                                value = x.at(h_sample_.unpad(i),
                                             w_sample_.unpad(j), k);
                            }
                            y.at(i_, j_, u, v, k) = value;
                        }
                    }
                }
            }
        }
    }
};
}  // namespace nn::ops
