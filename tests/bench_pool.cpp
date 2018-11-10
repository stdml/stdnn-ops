#include "benchmark.hpp"

#include <nn/ops>

template <int d1, int d2, int d3, int k, int p, int s, typename image_order,
          typename pool_algo>
struct bench_pool {
    static void run(benchmark::State &state)
    {
        using pool = nn::ops::pool<pool_algo, image_order>;
        const auto op = pool(pool::ksize(k, k), pool::stride(s, s));

        ttl::tensor<float, 4> x(1, d1, d2, d3);
        ttl::tensor<float, 4> y(op(x.shape()));

        for (auto _ : state) { op(ref(y), view(x)); }
    }
};

static void bench_max_pool_2x2_valid_chw_64_224_224(benchmark::State &state)
{
    bench_pool<64, 224, 224, 2, 0, 2, nn::ops::nchw, nn::ops::pool_max>::run(
        state);
}
BENCHMARK(bench_max_pool_2x2_valid_chw_64_224_224)
    ->Unit(benchmark::kMillisecond);

static void bench_max_pool_2x2_valid_hwc_224_224_64(benchmark::State &state)
{
    bench_pool<224, 224, 64, 2, 0, 2, nn::ops::nhwc, nn::ops::pool_max>::run(
        state);
}
BENCHMARK(bench_max_pool_2x2_valid_hwc_224_224_64)
    ->Unit(benchmark::kMillisecond);

static void bench_max_pool_3x3_same_chw_19_256_384(benchmark::State &state)
{
    bench_pool<19, 256, 384, 3, 1, 1, nn::ops::nchw, nn::ops::pool_max>::run(
        state);
}
BENCHMARK(bench_max_pool_3x3_same_chw_19_256_384)
    ->Unit(benchmark::kMillisecond);

static void bench_max_pool_3x3_same_hwc_256_384_19(benchmark::State &state)
{
    bench_pool<256, 384, 19, 3, 1, 1, nn::ops::nhwc, nn::ops::pool_max>::run(
        state);
}
BENCHMARK(bench_max_pool_3x3_same_hwc_256_384_19)
    ->Unit(benchmark::kMillisecond);
