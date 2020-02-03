#pragma once
#include <algorithm>

#include <ttl/device>
#include <ttl/nn/bits/engines/linag.hpp>
#include <ttl/nn/bits/kernels/blas.hpp>
#include <ttl/tensor>

namespace ttl::nn::kernels
{
template <typename R>
class axpy<host_memory, engines::builtin, R>
{
    using D = host_memory;
    static constexpr rank_t r = 1;

  public:
    void operator()(const tensor_ref<R, r, D> &z, const tensor_view<R, 0, D> &a,
                    const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        std::transform(
            x.data(), x.data_end(), y.data(), z.data(),
            [a = a.data()[0]](const R &x, const R &y) { return a * x + y; });
    }
};

template <typename R, typename E>
class mm<host_memory, E, R>
{
    using D = host_memory;

  public:
    void operator()(const tensor_ref<R, 2, D> &z, const tensor_view<R, 2, D> &x,
                    const tensor_view<R, 2, D> &y) const
    {
        engines::linag<E>::mm(x, y, z);
    }
};

template <typename R, typename E>
class mtm<host_memory, E, R>
{
    using D = host_memory;

  public:
    void operator()(const tensor_ref<R, 2, D> &z, const tensor_view<R, 2, D> &x,
                    const tensor_view<R, 2, D> &y) const
    {
        engines::linag<E>::mtm(x, y, z);
    }
};

template <typename R, typename E>
class mmt<host_memory, E, R>
{
    using D = host_memory;

  public:
    void operator()(const tensor_ref<R, 2, D> &z, const tensor_view<R, 2, D> &x,
                    const tensor_view<R, 2, D> &y) const
    {
        engines::linag<E>::mmt(x, y, z);
    }
};
}  // namespace ttl::nn::kernels
