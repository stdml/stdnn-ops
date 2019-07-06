#pragma once
#include <cmath>

#include <random>

#include <ttl/algorithm>

#include <nn/common.hpp>

namespace nn::ops
{
class zeros
{
  public:
    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &x) const
    {
        ttl::fill(x, static_cast<R>(0));
    }
};

class ones
{
  public:
    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &x) const
    {
        ttl::fill(x, static_cast<R>(1));
    }
};

template <typename R> class constant
{
    const R value_;

  public:
    constant(const R &value) : value_(value) {}

    template <ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &x) const
    {
        ttl::fill(x, value_);
    }
};

class uniform_distribution
{
  public:
    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &x) const
    {
        static_assert(std::is_floating_point<R>::value);
        const R value = static_cast<R>(1) / static_cast<R>(x.shape().size());
        ttl::fill(x, value);
    }
};

template <typename R> class truncated_normal
{
    const R stddev_;
    const R bound_;

  public:
    truncated_normal(const R &stddev)
        : stddev_(stddev), bound_(2 * std::fabs(stddev))
    {
    }

    template <ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &x) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<R> d(0, stddev_);
        std::generate(x.data(), x.data_end(), [&]() {
            R v = d(gen);
            while (std::fabs(v) > bound_) { v = d(gen); }
            return v;
        });
    }
};
}  // namespace nn::ops
