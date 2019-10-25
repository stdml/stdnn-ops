#pragma once
#include <nn/bits/traits/filter.hpp>
#include <nn/bits/traits/padding.hpp>

namespace nn
{
namespace traits
{
class conv1d_trait
{
  protected:
    using N = std::int32_t;

    using filter_t = linear_filter_trait<N>;
    using padding_t = linear_padding_trait<N>;

    const N stride_;
    const N rate_;
    const padding_t padding_;

  public:
    conv1d_trait(const int stride = 1, const int rate = 1,
                 const padding_t &padding = padding_t(0))
        : stride_(stride), rate_(rate), padding_(padding)
    {
    }
};
}  // namespace traits

namespace ops
{
class conv1d : public traits::conv1d_trait
{
    using conv1d_trait::conv1d_trait;

  public:
    ttl::shape<1> operator()(const ttl::shape<1> &x,
                             const ttl::shape<1> &y) const
    {
        const auto [n] = x.dims();
        const auto [k] = y.dims();
        return ttl::shape<1>(filter_t(k, stride_, rate_)(n, padding_));
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &z,
                    const ttl::tensor_view<R, 1> &x,
                    const ttl::tensor_view<R, 1> &y) const;
};
}  // namespace ops
}  // namespace nn
