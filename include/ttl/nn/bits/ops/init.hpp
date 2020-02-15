#pragma once
#include <cmath>

#include <random>

#include <ttl/algorithm>
#include <ttl/nn/bits/kernels/cpu/init.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class zeros
{
  public:
    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &x) const
    {
        kernels::zeros<D, R>()(flatten(x));
    }
};

class ones
{
  public:
    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &x) const
    {
        kernels::ones<D, R>()(flatten(x));
    }
};

template <typename R>
class constant
{
    const R value_;

  public:
    constant(const R &value) : value_(value) {}

    template <rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &x) const
    {
        (kernels::constant<D, R>(value_))(flatten(x));
    }
};

class uniform_constant
{
  public:
    template <typename R, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &x) const
    {
        kernels::uniform_constant<D, R>()(flatten(x));
    }
};

class truncated_normal
{
    const double stddev_;
    const double bound_;

    template <typename R, ttl::rank_t r, typename Engine>
    void operator()(const ttl::tensor_ref<R, r> &y, Engine &engine) const
    {
        std::normal_distribution<R> d(0, stddev_);
        std::generate(y.data(), y.data_end(), [&]() {
            R v = d(engine);
            while (std::fabs(v) > bound_) { v = d(engine); }
            return v;
        });
    }

  public:
    truncated_normal(const double &stddev)
        : stddev_(stddev), bound_(2 * std::fabs(stddev))
    {
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y) const
    {
        static_assert(std::is_floating_point<R>::value, "");
        std::random_device rd;
        std::mt19937 gen(rd());
        (*this)(y, gen);
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<uint32_t, 0> &x) const
    {
        static_assert(std::is_floating_point<R>::value, "");
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(x.data()[0]);
        (*this)(y, gen);
    }
};
}  // namespace ttl::nn::ops
