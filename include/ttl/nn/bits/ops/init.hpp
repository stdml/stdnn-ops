#pragma once
#include <cmath>

#include <random>

#include <ttl/algorithm>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
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
