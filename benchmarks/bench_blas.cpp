#include <ttl/device>
#include <ttl/nn/bits/ops/blas.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int l, int m, int n>
struct bench_mm {
    static void run(benchmark::State &state)
    {
        using R = float;
        using F = ttl::nn::ops::matmul;
        using B = bench<F, R, ttl::shape<2>, ttl::shape<2>, ttl::shape<2>>;
        B b(F(), ttl::make_shape(l, m), ttl::make_shape(m, n));
        b.template init<0>(ttl::nn::ops::ones());
        b.template init<1>(ttl::nn::ops::ones());
        run_bench(state, b);
    }
};

static void bench_mm_1024_1024_1024(benchmark::State &state)
{
    bench_mm<1024, 1024, 1024>::run(state);
}
BENCHMARK(bench_mm_1024_1024_1024)->Unit(benchmark::kMillisecond);

static void bench_mm_67600_9_32(benchmark::State &state)
{
    bench_mm<67600, 9, 32>::run(state);
}
BENCHMARK(bench_mm_67600_9_32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
