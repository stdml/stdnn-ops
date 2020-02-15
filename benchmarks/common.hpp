#pragma once
#include <chrono>
#include <tuple>

#include <ttl/tensor>

namespace ttl
{
template <typename Tuple, size_t... I>
auto view_tuple(const Tuple &tup, std::index_sequence<I...>)
{
    return std::make_tuple(view(std::get<I>(tup))...);
}

template <typename Tuple, size_t... I>
size_t tuple_data_size(const Tuple &tup, std::index_sequence<I...>)
{
    return (... + std::get<I>(tup).data_size());
}
}  // namespace ttl

template <typename F, typename R, typename S0, typename... Ss>
class bench
{
    static constexpr auto arity = sizeof...(Ss);

    using Y = ttl::tensor<R, S0::rank>;
    using Xs = std::tuple<ttl::tensor<R, Ss::rank>...>;

    F f;
    Y y;
    Xs xs;

  public:
    bench(const F &f, const Ss &... s)
        : f(f), y(f(s...)), xs(ttl::tensor<R, Ss::rank>(s)...)
    {
    }

    template <int i>
    const auto &input() const
    {
        return std::get<i>(xs);
    }

    template <int i, typename Init>
    void init(const Init &f) const
    {
        f(ttl::ref(input<i>()));
    }

    size_t data_size() const
    {
        return y.data_size() +
               ttl::tuple_data_size(xs, std::make_index_sequence<arity>());
    }

    void operator()() const
    {
        const auto args = std::tuple_cat(
            std::make_tuple(ttl::ref(y)),
            ttl::view_tuple(xs, std::make_index_sequence<arity>()));
        std::apply(f, args);
    }
};

class stop_watch
{
    using clock_t = std::chrono::high_resolution_clock;
    using instant_t = std::chrono::time_point<clock_t>;

    instant_t t0;

  public:
    using duratio_t = std::chrono::duration<double>;

    stop_watch() : t0(clock_t::now()) {}

    duratio_t stop()
    {
        const auto t1 = clock_t::now();
        const auto d = t1 - t0;
        t0 = t1;
        return d;
    }
};

double data_rate(const size_t size, const stop_watch::duratio_t d)
{
    constexpr auto Gi = 1 << 30;
    return size / d.count() / Gi;
}

#include <benchmark/benchmark.h>

template <typename B>
void run_bench(benchmark::State &state, B &b)
{
    stop_watch watch;
    int cnt = 0;
    for (auto _ : state) {
        ++cnt;
        b();
    }
    const auto ds = b.data_size();
    const auto rate = data_rate(ds, watch.stop() / cnt);
    printf("data size: %8d, data rate: %6.3f GiB/s (%d times)\n",
           static_cast<int>(ds), rate, cnt);
}
