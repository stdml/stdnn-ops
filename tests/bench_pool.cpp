#include "benchmark.hpp"

#include <nn/ops>

static void bench_same_max_pool_3x3(benchmark::State &state)
{
    int c = 19;
    int h = 256;
    int w = 384;
    ttl::tensor<float, 4> x(1, c, h, w);
    ttl::tensor<float, 4> y(x.shape());

    using pool = nn::ops::pool<nn::ops::pool_max, nn::ops::nchw>;
    const auto op =
        pool(pool::ksize(3, 3), pool::padding(1, 1), pool::stride(1, 1));

    for (auto _ : state) { op(ref(y), view(x)); }
}
BENCHMARK(bench_same_max_pool_3x3)->Unit(benchmark::kMillisecond);
