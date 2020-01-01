#include <ttl/nn/ops>

//
#include <ttl/nn/bits/ops/impl/conv1d.hpp>

namespace ttl::nn::ops
{
template void conv1d::operator()(const ttl::tensor_ref<float, 1> &z,
                                 const ttl::tensor_view<float, 1> &x,
                                 const ttl::tensor_view<float, 1> &y) const;

template void conv<nhwc>::operator()(const ttl::tensor_ref<float, 4> &z,
                                     const ttl::tensor_view<float, 4> &x,
                                     const ttl::tensor_view<float, 4> &y) const;

}  // namespace ttl::nn::ops
