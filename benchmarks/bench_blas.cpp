#include <ttl/device>
#include <ttl/nn/bits/ops/blas.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int l, int m, int n, typename E = ttl::nn::engines::cblas>
struct bench_mm {
    static void run(benchmark::State &state)
    {
        using R = float;
        using F = ttl::nn::kernels::mm<ttl::host_memory, E, R>;
        using S2 = ttl::shape<2>;
        using B = kernel_bench<F, R, S2, S2, S2>;
        B b(F(), S2(l, n), S2(l, m), S2(m, n));
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

static void bench_mm_67600_9_32_plain(benchmark::State &state)
{
    bench_mm<67600, 9, 32, ttl::nn::engines::builtin>::run(state);
}
BENCHMARK(bench_mm_67600_9_32_plain)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
