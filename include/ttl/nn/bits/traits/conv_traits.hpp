#pragma once
#include <ttl/nn/bits/traits/basic_traits.hpp>
#include <ttl/nn/bits/traits/multi_linear_sample.hpp>

namespace ttl::nn::traits
{
template <typename image_order>
class im2col_trait;

template <>
class im2col_trait<hw> : public multi_linear_sample_trait<2, uint32_t>
{
    using multi_linear_sample_trait::multi_linear_sample_trait;
};
}  // namespace ttl::nn::traits
