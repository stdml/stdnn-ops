#include <ttl/nn/bits/ops/im2col.hpp>

#include "benchmark.hpp"

template <int h, int w, int c> struct bench_im2col {
    using F =
        ttl::nn::ops::im2col<ttl::nn::traits::hwc, ttl::nn::traits::hwrsc>;

    static void run(benchmark::State &state)
    {
        F f(F::ksize(3, 3), F::padding(1, 1));
        ttl::tensor<float, 3> x(h, w, c);
        ttl::tensor<float, 5> y(f(x.shape()));
        for (auto _ : state) { f(ttl::ref(y), ttl::view(x)); }
    }
};

static void bench_im2col_hwc_hwrsc_256_384_32(benchmark::State &state)
{
    bench_im2col<256, 384, 32>::run(state);
}
BENCHMARK(bench_im2col_hwc_hwrsc_256_384_32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
