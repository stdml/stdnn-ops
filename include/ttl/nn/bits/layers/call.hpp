#pragma once
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
template <typename T, typename Op, typename... Ts>
T *new_result(const Op &op, const Ts &... args)
{
    auto y = new T(op(args.shape()...));
    op(ref(*y), args...);
    return y;
}

template <typename T, typename Init>
T *new_parameter(const typename T::shape_type &shape, const Init &init)
{
    auto w = new T(shape);
    init(ref(*w));
    return w;
}
}  // namespace ttl::nn::ops
