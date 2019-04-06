#pragma once
#include <algorithm>

#include <stdtensor>

#include <nn/common.hpp>

namespace nn::ops
{

struct relu {
    template <typename R> R operator()(R x) { return x > 0 ? x : 0.0; }
};

template <typename F> class pointwise
{
  public:
    template <ttl::rank_t r> shape<r> operator()(const shape<r> &x) const
    {
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<R, r> &x) const
    {
        const auto n = x.shape().size();
        std::transform(x.data(), x.data() + n, y.data(), F());
    }
};

}  // namespace nn::ops
