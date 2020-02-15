#include <ttl/nn/bits/ops/elementary.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int m, int n>
struct bench_add {
    static void run(benchmark::State &state)
    {
        using F = ttl::nn::ops::add;
        using B = bench<F, float, ttl::shape<2>, ttl::shape<2>, ttl::shape<2>>;
        B b(F(), ttl::make_shape(m, n), ttl::make_shape(m, n));
        b.init<0>(ttl::nn::ops::ones());
        b.init<1>(ttl::nn::ops::ones());
        run_bench(state, b);
    }
};

static void bench_add_1024_1024(benchmark::State &state)
{
    bench_add<1024, 1024>::run(state);
}
BENCHMARK(bench_add_1024_1024)->Unit(benchmark::kMillisecond);

static void bench_add_32_1024(benchmark::State &state)
{
    bench_add<32, 1024>::run(state);
}
BENCHMARK(bench_add_32_1024)->Unit(benchmark::kMicrosecond);

static void bench_add_1_1024(benchmark::State &state)
{
    bench_add<1, 1024>::run(state);
}
BENCHMARK(bench_add_1_1024)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
