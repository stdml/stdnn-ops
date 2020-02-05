#pragma once
#include <ttl/nn/bits/kernels/cpu/pool.hpp>
#include <ttl/nn/bits/ops/pool2d_trait.hpp>
#include <ttl/nn/bits/traits/multi_linear_sample.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
template <typename pool_algo, typename image_order>
class pool;

template <typename pool_algo>
class pool<pool_algo, hw> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

    using pool_trait::operator();

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 2, D> &y,
                    const tensor_view<R, 2, D> &x) const
    {
        const pool_trait<hw> &trait = *this;
        (kernels::pool<D, hw, pool_algo, R>(trait))(y, x);
    }
};

template <typename pool_algo>
class pool<pool_algo, hwc> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

    template <typename R, typename D>
    void operator()(const tensor_ref<R, 3, D> &y,
                    const tensor_view<R, 3, D> &x) const
    {
        const pool_trait<hw> &trait = *this;
        (kernels::pool<D, hwc, pool_algo, R>(trait))(y, x);
    }
};  // namespace ttl::nn::ops

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
