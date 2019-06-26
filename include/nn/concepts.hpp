#pragma once
#include <ttl/tensor>

namespace nn::internal
{
// TODO: make them concepts

template <typename R, ttl::rank_t r, ttl::rank_t r1> struct unary_operator {
    virtual void operator()(const ttl::tensor_ref<R, r> &,
                            const ttl::tensor_view<R, r1> &) = 0;
};

template <typename R, ttl::rank_t r, ttl::rank_t r1, ttl::rank_t r2>
struct binary_operator {
    virtual void operator()(const ttl::tensor_ref<R, r> &,
                            const ttl::tensor_view<R, r1> &,
                            const ttl::tensor_view<R, r2> &) = 0;
};

}  // namespace nn::internal
