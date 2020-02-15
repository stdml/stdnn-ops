#include <ttl/nn/bits/ops/init.hpp>
#include <ttl/nn/bits/ops/pool.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int d1, int d2, int d3, int k, int p, int s, typename image_order,
          typename pool_algo>
struct bench_pool {
    static void run(benchmark::State &state)
    {
        using F = ttl::nn::ops::pool<pool_algo, image_order>;
        const F op(F::ksize(k, k), F::padding(p, p), F::stride(s, s));
        using B = bench<F, float, ttl::shape<4>, ttl::shape<4>>;
        B b(op, ttl::make_shape(1, d1, d2, d3));
        // FIXME:  missing 'template' keyword prior to dependent template name
        // 'init'
        b.template init<0>(ttl::nn::ops::ones());
        run_bench(state, b);
    }
};

static void bench_max_pool_2x2_valid_chw_64_224_224(benchmark::State &state)
{
    bench_pool<64, 224, 224, 2, 0, 2, ttl::nn::ops::nchw,
               ttl::nn::ops::pool_max>::run(state);
}
BENCHMARK(bench_max_pool_2x2_valid_chw_64_224_224)
    ->Unit(benchmark::kMillisecond);

static void bench_max_pool_2x2_valid_hwc_224_224_64(benchmark::State &state)
{
    bench_pool<224, 224, 64, 2, 0, 2, ttl::nn::ops::nhwc,
               ttl::nn::ops::pool_max>::run(state);
}
BENCHMARK(bench_max_pool_2x2_valid_hwc_224_224_64)
    ->Unit(benchmark::kMillisecond);

static void bench_max_pool_3x3_same_chw_19_256_384(benchmark::State &state)
{
    bench_pool<19, 256, 384, 3, 1, 1, ttl::nn::ops::nchw,
               ttl::nn::ops::pool_max>::run(state);
}
BENCHMARK(bench_max_pool_3x3_same_chw_19_256_384)
    ->Unit(benchmark::kMillisecond);

static void bench_max_pool_3x3_same_hwc_256_384_19(benchmark::State &state)
{
    bench_pool<256, 384, 19, 3, 1, 1, ttl::nn::ops::nhwc,
               ttl::nn::ops::pool_max>::run(state);
}
BENCHMARK(bench_max_pool_3x3_same_hwc_256_384_19)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
