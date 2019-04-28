#pragma once
#include <algorithm>

#include <nn/common.hpp>

namespace nn::ops
{
class identity
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
        std::copy(x.data(), x.data() + x.shape().size(), y.data());
    }
};

struct scalar_add {
    template <typename R> R operator()(const R &x, const R &y) const
    {
        return x + y;
    }
};

struct scalar_sub {
    template <typename R> R operator()(const R &x, const R &y) const
    {
        return x - y;
    }
};

struct scalar_mul {
    template <typename R> R operator()(const R &x, const R &y) const
    {
        return x * y;
    }
};

template <typename F> class _binary_pointwise
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert(x == y);
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
        std::transform(x.data(), x.data() + x.shape().size(), y.data(),
                       z.data(), F());
    }
};

using add = _binary_pointwise<scalar_add>;
using sub = _binary_pointwise<scalar_sub>;

}  // namespace nn::ops
