#include "benchmark.hpp"

#include <nn/ops>

template <int c, int h, int w> struct bench_same_max_pool_3x3 {
    static void run(benchmark::State &state)
    {
        ttl::tensor<float, 4> x(1, c, h, w);
        ttl::tensor<float, 4> y(x.shape());

        using pool = nn::ops::pool<nn::ops::pool_max, nn::ops::nchw>;
        const auto op =
            pool(pool::ksize(3, 3), pool::padding(1, 1), pool::stride(1, 1));

        for (auto _ : state) { op(ref(y), view(x)); }
    }
};

static void bench_same_max_pool_3x3_chw_19_256_384(benchmark::State &state)
{
    bench_same_max_pool_3x3<19, 256, 384>::run(state);
}
BENCHMARK(bench_same_max_pool_3x3_chw_19_256_384)
    ->Unit(benchmark::kMillisecond);
