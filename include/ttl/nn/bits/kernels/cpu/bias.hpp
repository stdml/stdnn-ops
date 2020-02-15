#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/bias.hpp>
#include <ttl/nn/bits/ops/elementary.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>
// #include <ttl/nn/kernels/macro>
#include <ttl/nn/traits>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename image_order, typename F, typename R>
class apply_bias<host_memory, image_order, F, R>
{
    static constexpr auto r = traits::rank_of<image_order>;
    static constexpr auto p = traits::bias_position<image_order>;

  public:
    void operator()(const tensor_ref<R, r> &z, const tensor_view<R, r> &x,
                    const tensor_view<R, 1> &y) const
    {  // FIXME: improve performance
        F f;
        const auto s = x.shape();
        // FIXME: reuse s.template coord<p>
        for (auto i : range(s.size())) {
            const auto j = s.template coord<p>(i);
            z.data()[i] = f(x.data()[i], y.data()[j]);
        }
    }

    template <typename... Ts>
    void operator()(const tensor_ref<R, r> &z, const tensor_view<R, r> &x,
                    const tensor_view<R, 1> &y0, const Ts &... ys) const
    {
        F f;
        const auto s = x.shape();
        // FIXME: reuse s.template coord<p>
        for (auto i : range(s.size())) {
            const auto j = s.template coord<p>(i);
            z.data()[i] = f(x.data()[i], y0.data()[j], ys.data()[j]...);
        }
    }
};

template <typename F, typename R>
class apply_bias<host_memory, traits::nhwc, F, R>
{
  public:
    void operator()(const tensor_ref<R, 2> &z, const tensor_view<R, 2> &x,
                    const tensor_view<R, 1> &y) const
    {
        for (auto i : range<0>(z)) {
            ttl::nn::kernels::host_binary_pointwise<F, R>()(z[i], x[i], y);
        }
    }

    void operator()(const tensor_ref<R, 4> &z, const tensor_view<R, 4> &x,
                    const tensor_view<R, 1> &y) const
    {
        operator()(ops::as_matrix<3, 1>(z), ops::as_matrix<3, 1>(x), y);
    }
};

// extern template class apply_bias<host_memory, traits::nhwc, ops::scalar_add,
//                                  float>;
// FOR_ALL_TYPES(DECLARE, class, apply_bias, host_memory, traits::nhwc,
//               ops::scalar_add);
}  // namespace ttl::nn::kernels
