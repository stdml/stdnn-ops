#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/pool.hpp>
#include <ttl/nn/bits/traits/pool_traits.hpp>
#include <ttl/tensor>

// FIXME: move out of ops
#include <ttl/nn/bits/ops/pool2d_trait.hpp>

namespace ttl::nn::kernels
{
namespace internal
{
template <typename R>
class max_accumulator
{
    R val;

  public:
    max_accumulator() : val(std::numeric_limits<R>::lowest()) {}

    void operator()(R x)
    {
        if (x > val) { val = x; }
    }

    operator R() const { return val; }
};

template <typename R>
class mean_accumulator
{
    R sum;
    uint32_t n;

  public:
    mean_accumulator() : sum(0), n(0) {}

    void operator()(R x)
    {
        sum += x;
        ++n;
    }

    operator R() const { return sum / n; }
};

template <typename pool_algo, typename R>
struct accumulator;

template <typename R>
struct accumulator<traits::pool_max, R> {
    using type = max_accumulator<R>;
};
template <typename R>
struct accumulator<traits::pool_mean, R> {
    using type = mean_accumulator<R>;
};
}  // namespace internal

template <typename R, typename algo>
class pool<host_memory, traits::hw, algo, R>
    : public ops::pool_trait<traits::hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

    //   TODO: support strided tensor
    void operator()(const tensor_ref<R, 2> &y, const tensor_view<R, 2> &x) const
    {
        using accumulator = typename internal::accumulator<algo, R>::type;

        const auto [h, w] = x.dims();
        const auto [r, s] = get_ksize().dims();

        for (auto i_ : range<0>(y)) {
            for (auto j_ : range<1>(y)) {
                accumulator acc;
                for (auto u : range(r)) {
                    for (auto v : range(s)) {
                        const auto i = h_sample_(i_, u);
                        const auto j = w_sample_(j_, v);
                        if (h_sample_.inside(i, h) && w_sample_.inside(j, w)) {
                            acc(x.at(h_sample_.unpad(i), w_sample_.unpad(j)));
                        }
                    }
                }
                y.at(i_, j_) = acc;
            }
        }
    }
};

template <typename R, typename algo>
class pool<host_memory, traits::hwc, algo, R>
    : public ops::pool_trait<traits::hw>
{
    using pool_trait::pool_trait;

  public:
    pool(const pool_trait &t) : pool_trait(t) {}

    void operator()(const tensor_ref<R, 3> &y, const tensor_view<R, 3> &x) const
    {
        using accumulator = typename internal::accumulator<algo, R>::type;

        const auto [h, w, c] = x.dims();
        // const auto [h_, w_, _c] = y.dims();
        const auto [r, s] = get_ksize().dims();
        contract_assert_eq(c, std::get<2>(y.dims()));

        for (auto k : range<2>(y)) {
            for (auto i_ : range<0>(y)) {
                for (auto j_ : range<1>(y)) {
                    accumulator acc;
                    for (auto u : range(r)) {
                        for (auto v : range(s)) {
                            const auto i = h_sample_(i_, u);
                            const auto j = w_sample_(j_, v);
                            if (h_sample_.inside(i, h) &&
                                w_sample_.inside(j, w)) {
                                acc(x.at(h_sample_.unpad(i), w_sample_.unpad(j),
                                         k));
                            }
                        }
                    }
                    y.at(i_, j_, k) = acc;
                }
            }
        }
    }
};
}  // namespace ttl::nn::kernels
