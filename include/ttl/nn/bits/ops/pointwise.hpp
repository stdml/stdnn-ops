#pragma once
#include <algorithm>

#include <ttl/tensor>

#include <ttl/nn/bits/ops/shape_algo.hpp>
#include <ttl/nn/common.hpp>

namespace nn::ops
{

struct relu {
    template <typename R> R operator()(R x) { return x > 0 ? x : 0.0; }
};

// TODO: leaky relu

template <typename F> class pointwise : public nn::ops::endofunction
{
    const F f_;

  public:
    using endofunction::operator();

    pointwise(const F &f = F()) : f_(f) {}

    template <typename R, typename R1, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<R1, r> &x) const
    {
        std::transform(x.data(), x.data_end(), y.data(), f_);
    }
};

}  // namespace nn::ops
