#pragma once
#include <ttl/nn/bits/kernels/elementary.hpp>
#include <ttl/nn/bits/ops/std_function.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class identity : public endofunction
{
  public:
    using endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &y,
                    const tensor_view<R, r, D> &x) const
    {
        kernels::identity<R, D>()(flatten(y), flatten(x));
    }
};

class noop
{
  public:
    template <typename T>
    void operator()(const T &y) const
    {
        // noop
    }

    template <typename S, typename T>
    void operator()(const T &y, const S &x) const
    {
        // noop
    }
};

struct scalar_add {
    template <typename R>
    R operator()(const R &x, const R &y) const
    {
        return x + y;
    }
};

struct scalar_sub {
    template <typename R>
    R operator()(const R &x, const R &y) const
    {
        return x - y;
    }
};

struct scalar_mul {
    template <typename R>
    R operator()(const R &x, const R &y) const
    {
        return x * y;
    }
};

template <typename F>
class _binary_pointwise : public binary_endofunction
{
  public:
    using binary_endofunction::operator();

    template <typename R, rank_t r>
    void operator()(const tensor_ref<R, r> &z, const tensor_view<R, r> &x,
                    const tensor_view<R, r> &y) const
    {
        std::transform(x.data(), x.data_end(), y.data(), z.data(), F());
    }
};

class add : public binary_endofunction
{
  public:
    using binary_endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &z, const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        kernels::add<R, D>()(flatten(z), flatten(x), flatten(y));
    }
};

class sub : public binary_endofunction
{
  public:
    using binary_endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &z, const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        kernels::sub<R, D>()(flatten(z), flatten(x), flatten(y));
    }
};

class mul : public binary_endofunction
{
  public:
    using binary_endofunction::operator();

    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &z, const tensor_view<R, r, D> &x,
                    const tensor_view<R, r, D> &y) const
    {
        kernels::mul<R, D>()(flatten(z), flatten(x), flatten(y));
    }
};
}  // namespace ttl::nn::ops
