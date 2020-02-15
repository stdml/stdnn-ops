#include <ttl/nn/bits/ops/conv2d.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int n, int h, int w, int c, int r, int s, int d>
struct bench_conv2d {
    static void run(benchmark::State &state)
    {
        using F = ttl::nn::ops::conv<ttl::nn::traits::nhwc,  //
                                     ttl::nn::traits::rscd>;
        using B = bench<F, float, ttl::shape<4>, ttl::shape<4>, ttl::shape<4>>;
        B b(F(), ttl::make_shape(n, h, w, c), ttl::make_shape(r, s, c, d));
        b.init<0>(ttl::nn::ops::ones());
        b.init<1>(ttl::nn::ops::ones());
        run_bench(state, b);
    }
};

static void bench_conv2d_nhwc_rscd_100_28_28_1_3_3_32(benchmark::State &state)
{
    bench_conv2d<100, 28, 28, 1, 3, 3, 32>::run(state);
}
BENCHMARK(bench_conv2d_nhwc_rscd_100_28_28_1_3_3_32)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
