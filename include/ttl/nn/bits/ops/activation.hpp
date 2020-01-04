#pragma once
#include <ttl/nn/bits/kernels/activation.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class softmax : public endofunction
{
  public:
    using endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &y, const tensor_view<R, r, D> &x,
                    R eps = static_cast<R>(1e-6)) const
    {
        (kernels::softmax<R, D>(eps))(as_matrix<r - 1, 1>(y),
                                      as_matrix<r - 1, 1>(x));
    }
};

struct relu_scalar {
    template <typename R>
    R operator()(R x)
    {
        return x > 0 ? x : 0.0;
    }
};

// TODO: leaky_relu

class relu : public endofunction
{
  public:
    using endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &y,
                    const tensor_view<R, r, D> &x) const
    {
        kernels::relu<R, D>()(flatten(y), flatten(x));
    }
};
}  // namespace ttl::nn::ops
