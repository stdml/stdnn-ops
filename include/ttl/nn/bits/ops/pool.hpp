#pragma once
#include <ttl/nn/bits/kernels/cpu/pool.hpp>
#include <ttl/nn/bits/ops/pool2d_trait.hpp>
#include <ttl/nn/bits/traits/multi_linear_sample.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
namespace internal
{
template <typename R>
class max_accumulator
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

template <typename R>
class mean_accumulator
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

template <typename pool_algo, typename R>
struct accumulator;

template <typename R>
struct accumulator<traits::pool_max, R> {
    using type = max_accumulator<R>;
};
template <typename R>
struct accumulator<traits::pool_mean, R> {
    using type = mean_accumulator<R>;
};
}  // namespace internal

template <typename pool_algo, typename image_order>
class pool;

template <typename pool_algo>
class pool<pool_algo, hw> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

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

        const auto [h, w] = x.shape().dims();
        const auto [h_, w_] = y.shape().dims();
        const auto [r, s] = get_ksize().dims();

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

template <typename pool_algo>
class pool<pool_algo, hwc> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 3> &y,
                    const ttl::tensor_view<R, 3> &x) const
    {
        using accumulator = typename internal::accumulator<pool_algo, R>::type;

        const auto [h, w, c] = x.shape().dims();
        const auto [h_, w_, _c] = y.shape().dims();
        const auto [r, s] = get_ksize().dims();
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
    pool(const pool_trait &t) : pool_trait(t) {}

    shape<4> operator()(const shape<4> &x) const
    {
        return pooled_shape<nhwc, pool_trait<hw>>(x, *this);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x) const
    {
        pool<pool_algo, hwc> op(get_ksize(), get_stride());
        for (auto i : range(x.shape().dims()[0])) { op(y[i], x[i]); }
    }
};

template <typename pool_algo>
class pool<pool_algo, nchw> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

    shape<4> operator()(const shape<4> &x) const
    {
        return pooled_shape<nchw, pool_trait<hw>>(x, *this);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x) const
    {
        pool<pool_algo, hw> op(get_ksize(), get_stride());
        for (auto i : range(x.shape().dims()[0])) {
            for (auto j : range(x.shape().dims()[1])) { op(y[i][j], x[i][j]); }
        }
    }
};
}  // namespace ttl::nn::ops
