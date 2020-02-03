#pragma once
#include <ttl/algorithm>
#include <ttl/nn/bits/kernels/cpu/elementary.hpp>
#include <ttl/nn/bits/kernels/cpu/reduce.hpp>
#include <ttl/nn/bits/ops/bias.hpp>
#include <ttl/nn/bits/ops/reduce.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops::grad
{
template <typename image_order, arity_t>
class add_bias;

template <typename image_order>
class add_bias<image_order, 0>
    : public basic_gradient_function<ops::add_bias<image_order>, 0>
{
    static constexpr auto r = traits::rank_of<image_order>;
    static constexpr auto p = traits::bias_position<image_order>;

    using P = basic_gradient_function<ops::add_bias<image_order>, 0>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void
    operator()(const tensor_ref<R, r, D> &gx, const tensor_view<R, r, D> &gz,
               const tensor_view<R, r, D> &z, const tensor_view<R, r, D> &x,
               const tensor_view<R, 1, D> &y) const
    {
        kernels::identity<D, R>()(flatten(gx), flatten(gz));
    }
};

template <>
class add_bias<traits::hw, 1>
    : public basic_gradient_function<ops::add_bias<traits::hw>, 1>
{
    using P = basic_gradient_function<ops::add_bias<traits::hw>, 1>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void
    operator()(const tensor_ref<R, 1, D> &gy, const tensor_view<R, 2, D> &gz,
               const tensor_view<R, 2, D> &z, const tensor_view<R, 2, D> &x,
               const tensor_view<R, 1, D> &y) const
    {
        kernels::outter_contraction<D, R>()(gy, gz);
    }
};

template <>
class add_bias<traits::nhwc, 1>
    : public basic_gradient_function<ops::add_bias<traits::nhwc>, 1>
{
    using P = basic_gradient_function<ops::add_bias<traits::nhwc>, 1>;
    using P::P;

  public:
    using P::operator();

    template <typename R, typename D>
    void
    operator()(const tensor_ref<R, 1, D> &gy, const tensor_view<R, 4, D> &gz,
               const tensor_view<R, 4, D> &z, const tensor_view<R, 4, D> &x,
               const tensor_view<R, 1, D> &y) const
    {
        kernels::outter_contraction<D, R>()(gy, ops::as_matrix<3, 1>(gz));
    }
};

// TODO: class add_bias<traits::nchw, 1>
// FIXME: unify add_bias<image_order, 1>
}  // namespace ttl::nn::ops::grad
