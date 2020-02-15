#include <ttl/nn/bits/ops/gradients/activation.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int n, int h, int w, int c>
struct bench_grad_relu {
    static void run(benchmark::State &state)
    {
        using F = ttl::nn::ops::relu;
        using G0 = ttl::nn::ops::grad::relu<0>;
        using S4 = ttl::shape<4>;
        using B = bench<G0, float, S4, S4, S4, S4>;
        B b(G0(F()), S4(n, h, w, c), S4(n, h, w, c), S4(n, h, w, c));
        b.init<0>(ttl::nn::ops::ones());
        b.init<1>(ttl::nn::ops::ones());
        run_bench(state, b);
    }
};

static void bench_grad_relu_100_26_26_32(benchmark::State &state)
{
    bench_grad_relu<100, 26, 26, 32>::run(state);
}
BENCHMARK(bench_grad_relu_100_26_26_32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
