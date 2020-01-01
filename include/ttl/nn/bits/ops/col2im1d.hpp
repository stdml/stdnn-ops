#pragma once
#include <ttl/nn/bits/traits/linear_sample.hpp>

namespace nn
{
namespace traits
{
class col2im_trait : public linear_sample_trait<int32_t>
{
    using linear_sample_trait::linear_sample_trait;
};
}  // namespace traits

namespace ops
{
class col2im1d : public traits::col2im_trait
{
    using col2im_trait::col2im_trait;
    using col2im_trait::operator();

  public:
    ttl::shape<1> operator()(const ttl::shape<2> &x) const
    {
        const auto m = std::get<0>(x.dims());
        int n = (m - 1) * filter_.stride() + filter_.patch_size();
        return ttl::shape<1>(n - padding_(0));
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &y,
                    const ttl::tensor_view<R, 2> &x) const;
};
}  // namespace ops
}  // namespace nn
