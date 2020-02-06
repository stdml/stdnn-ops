#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/bias.hpp>
#include <ttl/nn/traits>
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
}  // namespace ttl::nn::kernels
