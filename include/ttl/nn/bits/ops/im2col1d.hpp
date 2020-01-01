#pragma once
#include <ttl/nn/bits/traits/linear_sample.hpp>

namespace nn
{
namespace traits
{
class im2col1d_trait : public linear_sample_trait<int32_t>
{
    using linear_sample_trait::linear_sample_trait;
};
}  // namespace traits

namespace ops
{
class im2col1d : public traits::im2col1d_trait
{
    using im2col1d_trait::im2col1d_trait;
    using im2col1d_trait::operator();

  public:
    ttl::shape<2> operator()(const ttl::shape<1> &x) const
    {
        const auto [n] = x.dims();
        const auto m = (*this)(n);
        return ttl::make_shape(m, get_ksize());
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &y,
                    const ttl::tensor_view<R, 1> &x) const;
};
}  // namespace ops
}  // namespace nn
