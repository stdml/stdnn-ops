#include <ttl/nn/bits/ops/col2im.hpp>
#include <ttl/nn/bits/ops/im2col.hpp>

#include "benchmark.hpp"

template <int h, int w, int c>
struct bench_image_to_column {
    using F = ttl::nn::ops::im2col<ttl::nn::traits::hwc,  //
                                   ttl::nn::traits::hwrsc>;
    using G = ttl::nn::ops::col2im<ttl::nn::traits::hwc,  //
                                   ttl::nn::traits::hwrsc>;

    static void run_im2col(benchmark::State &state)
    {
        F f(F::ksize(3, 3), F::padding(1, 1));
        using R = float;
        ttl::tensor<R, 3> x(h, w, c);
        ttl::tensor<R, 5> y(f(x.shape()));
        ttl::fill(ttl::ref(x), static_cast<R>(1));
        ttl::fill(ttl::ref(y), static_cast<R>(1));
        for (auto _ : state) { f(ttl::ref(y), ttl::view(x)); }
    }

    static void run_col2im(benchmark::State &state)
    {
        F f(F::ksize(3, 3), F::padding(1, 1));
        G g(G::ksize(3, 3), G::padding(1, 1));
        using R = float;
        ttl::tensor<R, 3> x(h, w, c);
        ttl::tensor<R, 5> y(f(x.shape()));
        ttl::fill(ttl::ref(x), static_cast<R>(1));
        ttl::fill(ttl::ref(y), static_cast<R>(1));
        for (auto _ : state) { g(ttl::ref(x), ttl::view(y)); }
    }
};

static void bench_im2col_hwc_hwrsc_256_384_32(benchmark::State &state)
{
    bench_image_to_column<256, 384, 32>::run_im2col(state);
}
BENCHMARK(bench_im2col_hwc_hwrsc_256_384_32)->Unit(benchmark::kMillisecond);

static void bench_col2im_hwc_hwrsc_256_384_32(benchmark::State &state)
{
    bench_image_to_column<256, 384, 32>::run_col2im(state);
}
BENCHMARK(bench_col2im_hwc_hwrsc_256_384_32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
