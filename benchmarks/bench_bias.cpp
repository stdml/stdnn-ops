#include <ttl/nn/bits/ops/bias.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int n, int h, int w, int c>
struct bench_bias {
    static void run(benchmark::State &state)
    {
        using F = ttl::nn::ops::add_bias<ttl::nn::traits::nhwc>;
        using B = bench<F, float, ttl::shape<4>, ttl::shape<4>, ttl::shape<1>>;
        B b(F(), ttl::make_shape(n, h, w, c), ttl::make_shape(c));
        b.init<0>(ttl::nn::ops::ones());
        b.init<1>(ttl::nn::ops::ones());
        for (auto _ : state) { b(); }
    }
};

static void bench_bias_nhwc_100_26_26_32(benchmark::State &state)
{
    bench_bias<100, 26, 26, 32>::run(state);
}
BENCHMARK(bench_bias_nhwc_100_26_26_32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
