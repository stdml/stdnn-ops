#pragma once

#include <nn/bits/ops/linear_sample.hpp>
#include <nn/bits/ops/traits.hpp>
#include <nn/common.hpp>

namespace nn::ops
{

template <typename image_order> class pool_trait;

template <> class pool_trait<hw>
{
  protected:
    struct ksize_trait;
    struct stride_trait;

    using dim_t = size_t;
    using sample1d_t_ = linear_sample_trait<dim_t>;

    using padding_1d_t = typename sample1d_t_::padding_t;
    using padding_t = std::array<padding_1d_t, 2>;
    using ksize_t = std::experimental::new_type<shape<2>, ksize_trait>;
    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;

    static constexpr auto default_ksize = ksize_t(2, 2);

    const sample1d_t_ h_sample_;
    const sample1d_t_ w_sample_;

    ksize_t get_ksize() const
    {
        return ksize_t(h_sample_.get_ksize(), w_sample_.get_ksize());
    }

    stride_t get_stride() const
    {
        return stride_t(h_sample_.get_stride(), w_sample_.get_stride());
    }

    padding_t get_padding() const
    {
        return padding(h_sample_.get_padding(), w_sample_.get_padding());
    }

  public:
    using sample1d_t = sample1d_t_;

    static padding_1d_t padding_1d(dim_t p) { return padding_1d_t(p, p); }

    static padding_1d_t padding_1d(dim_t left, dim_t right)
    {
        return padding_1d_t(left, right);
    }

    static padding_t padding(const padding_1d_t &r, const padding_1d_t &s)
    {
        return {r, s};
    };

    static padding_t padding(dim_t r, dim_t s)
    {
        return padding(sample1d_t::padding(r), sample1d_t::padding(s));
    }

    static ksize_t ksize(dim_t r, dim_t s) { return ksize_t(r, s); }

    static stride_t stride(dim_t r, dim_t s) { return stride_t(r, s); }

    pool_trait() : pool_trait(default_ksize) {}

    pool_trait(const ksize_t &ksize)
        : pool_trait(ksize, stride(ksize.dims[0], ksize.dims[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const stride_t &stride)
        : pool_trait(ksize, padding(0, 0), stride)
    {
    }

    pool_trait(const ksize_t &ksize, const padding_t &padding)
        : pool_trait(ksize, padding, stride(ksize.dims[0], ksize.dims[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const padding_t &padding,
               const stride_t &stride)
        : h_sample_(ksize.dims[0], stride.dims[0], 1, std::get<0>(padding)),
          w_sample_(ksize.dims[1], stride.dims[1], 1, std::get<1>(padding))
    {
    }

    shape<2> operator()(const shape<2> &x) const
    {
        return shape<2>(h_sample_(x.dims[0]), w_sample_(x.dims[1]));
    }
};

struct pool_max;
struct pool_mean;

namespace internal
{
template <typename R> class max_accumulator
{
    R val;

  public:
    max_accumulator() : val(std::numeric_limits<R>::lowest()) {}

    void operator()(R x)
    {
        if (x > val) { val = x; }
    }

    operator R() const { return val; }
};

template <typename R> class mean_accumulator
{
    R sum;
    uint32_t n;

  public:
    mean_accumulator() : sum(0), n(0) {}

    void operator()(R x)
    {
        sum += x;
        ++n;
    }

    operator R() const { return sum / n; }
};

template <typename pool_algo, typename R> struct accumulator;

template <typename R> struct accumulator<pool_max, R> {
    using type = max_accumulator<R>;
};
template <typename R> struct accumulator<pool_mean, R> {
    using type = mean_accumulator<R>;
};
}  // namespace internal

template <typename pool_algo, typename image_order> class pool;

template <typename pool_algo> class pool<pool_algo, hw> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    shape<2> operator()(const shape<2> &x) const
    {
        return pool_trait<hw>::operator()(x);  // FIXME: auto pass through
    }

    //   TODO: support strided tensor
    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &y,
                    const ttl::tensor_view<R, 2> &x) const
    {
        using accumulator = typename internal::accumulator<pool_algo, R>::type;

        const auto [h, w] = x.shape().dims;
        const auto [h_, w_] = y.shape().dims;
        const auto [r, s] = get_ksize().dims;

        for (auto i_ : range(h_)) {
            for (auto j_ : range(w_)) {
                accumulator acc;
                for (auto u : range(r)) {
                    for (auto v : range(s)) {
                        const auto i = h_sample_(i_, u);
                        const auto j = w_sample_(j_, v);
                        if (h_sample_.inside(i, h) && w_sample_.inside(j, w)) {
                            acc(x.at(h_sample_.unpad(i), w_sample_.unpad(j)));
                        }
                    }
                }
                y.at(i_, j_) = acc;
            }
        }
    }
};

template <typename pool_algo> class pool<pool_algo, hwc> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    template <typename R>
    void operator()(const ttl::tensor_ref<R, 3> &y,
                    const ttl::tensor_view<R, 3> &x) const
    {
        using accumulator = typename internal::accumulator<pool_algo, R>::type;

        const auto [h, w, c] = x.shape().dims;
        const auto [h_, w_, _c] = y.shape().dims;
        const auto [r, s] = get_ksize().dims;
        contract_assert_eq(c, _c);

        for (auto k : range(c)) {
            for (auto i_ : range(h_)) {
                for (auto j_ : range(w_)) {
                    accumulator acc;
                    for (auto u : range(r)) {
                        for (auto v : range(s)) {
                            const auto i = h_sample_(i_, u);
                            const auto j = w_sample_(j_, v);
                            if (h_sample_.inside(i, h) &&
                                w_sample_.inside(j, w)) {
                                acc(x.at(h_sample_.unpad(i), w_sample_.unpad(j),
                                         k));
                            }
                        }
                    }
                    y.at(i_, j_, k) = acc;
                }
            }
        }
    }
};

template <typename order, typename PoolTrait>
shape<4> pooled_shape(const shape<4> &x, const PoolTrait &pt)
{
    return batched_image_shape<order>(batch_size<order>(x),
                                      pt(image_shape<order>(x)),
                                      channel_size<order>(x));
}

template <typename pool_algo>
class pool<pool_algo, nhwc> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    shape<4> operator()(const shape<4> &x) const
    {
        return pooled_shape<nhwc, pool_trait<hw>>(x, *this);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x) const
    {
        pool<pool_algo, hwc> op(get_ksize(), get_stride());
        for (auto i : range(x.shape().dims[0])) { op(y[i], x[i]); }
    }
};

template <typename pool_algo>
class pool<pool_algo, nchw> : public pool_trait<hw>
{
  public:
    using pool_trait::pool_trait;

    shape<4> operator()(const shape<4> &x) const
    {
        return pooled_shape<nchw, pool_trait<hw>>(x, *this);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x) const
    {
        pool<pool_algo, hw> op(get_ksize(), get_stride());
        for (auto i : range(x.shape().dims[0])) {
            for (auto j : range(x.shape().dims[1])) { op(y[i][j], x[i][j]); }
        }
    }
};

}  // namespace nn::ops
