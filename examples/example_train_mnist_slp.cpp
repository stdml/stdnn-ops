#include <cstdlib>

#include <ttl/algorithm>
#include <ttl/nn/bits/ops/gradients/softmax.hpp>
#include <ttl/nn/bits/ops/gradients/xentropy.hpp>
#include <ttl/nn/experimental/datasets>
#include <ttl/nn/layers>
#include <ttl/nn/ops>
#include <ttl/range>

#include "trace.hpp"
#include "utils.hpp"

class slp
{
  public:
    ttl::shape<2> operator()(const ttl::shape<2> &x, const ttl::shape<2> &y,
                             const ttl::shape<1> &z) const
    {
        const auto [n, m] = x.dims();
        const auto [_m, k] = y.dims();
        contract_assert_eq(m, _m);
        contract_assert_eq(k, std::get<0>(z.dims()));
        return ttl::shape<2>(n, k);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &ys,
                    const ttl::tensor_view<R, 2> &xs,
                    const ttl::tensor_view<R, 2> &w,
                    const ttl::tensor_view<R, 1> &b)
    {
        ttl::nn::ops::matmul()(ys, xs, w);
        ttl::nn::ops::add_bias<ttl::nn::ops::hw>()(ys, view(ys), b);
    }

    template <typename R>
    void
    grad(const ttl::tensor_ref<R, 2> &g_w, const ttl::tensor_ref<R, 1> &g_b,
         const ttl::tensor_view<R, 2> &g_ys, const ttl::tensor_view<R, 2> &ys,
         const ttl::tensor_view<R, 2> &xs, const ttl::tensor_view<R, 2> &w,
         const ttl::tensor_view<R, 1> &b)
    {
        for (auto l : ttl::range<1>(g_ys)) {
            R tot = 0;
            for (auto i : ttl::range<0>(g_ys)) { tot += g_ys.at(i, l); }
            g_b.at(l) = tot;
        }
        ttl::nn::engines::linag<ttl::nn::engines::default_engine>::mtm(xs, g_ys,
                                                                       g_w);
    }
};

template <typename R>
void loss(const ttl::tensor_ref<R, 0> &l, const ttl::tensor_view<R, 2> &ys,
          const ttl::tensor_view<R, 2> &y_s)
{
    ttl::tensor<R, 1> ls(ys.shape().dims()[0]);
    ttl::nn::ops::xentropy()(ref(ls), y_s, ys);
    ttl::nn::ops::mean()(ref(l), view(ls));
}

template <typename R, typename R1>
R accuracy(const ttl::tensor_view<R1, 2> &ys,
           const ttl::tensor_view<R1, 2> &y_s)
{
    const ttl::nn::ops::argmax argmax;
    const ttl::tensor<uint32_t, 1> preditions(argmax(ys.shape()));
    const ttl::tensor<uint32_t, 1> labels(argmax(y_s.shape()));
    argmax(ref(preditions), ys);
    argmax(ref(labels), y_s);
    const ttl::tensor<float, 0> sim;
    ttl::nn::ops::similarity()(ref(sim),  //
                               view(preditions), view(labels));
    return sim.data()[0];
}

template <typename D, typename R>
void load_data(const D &ds, int offset, int batch_size,
               const ttl::tensor_ref<R, 2> &xs,
               const ttl::tensor_ref<R, 2> &y_s)
{
    const auto f = [](uint8_t p) { return p / 255.0; };
    ttl::nn::ops::pointwise<decltype(f)> normalize_pixel(f);
    normalize_pixel(ttl::tensor_ref<R, 3>(xs.data(), batch_size, 28,
                                          28) /* FIXME: use reshape */,
                    view(ds.images.slice(offset, offset + batch_size)));
    ttl::nn::ops::onehot(10)(
        y_s, view(ds.labels.slice(offset, offset + batch_size)));
}

template <typename D, typename R>
void train_slp_model(const D &ds,  //
                     const ttl::tensor_ref<R, 2> &w,
                     const ttl::tensor_ref<R, 1> &b,  //
                     const int batch_size = 100)
{
    const auto [n, height, width] = ds.images.shape().dims();
    const int k = 10;

    ttl::tensor<R, 2> xs(batch_size, height * width);
    ttl::tensor<R, 2> zs(batch_size, k);   // zs = xs * w + b
    ttl::tensor<R, 2> ys(batch_size, k);   // ys = softmax(zs)
    ttl::tensor<R, 2> y_s(batch_size, k);  // labels
    ttl::tensor<R, 1> ls(batch_size);      // xentropy(y_s, ys)
    ttl::tensor<R, 0> l;                   // l = mean(ls)

    ttl::tensor<R, 0> g_l;
    ttl::tensor<R, 1> g_ls(batch_size);
    ttl::tensor<R, 2> g_ys(ys.shape());
    ttl::tensor<R, 2> g_zs(zs.shape());
    ttl::tensor<R, 2> g_w(w.shape());
    ttl::tensor<R, 1> g_b(b.shape());

    g_l.data()[0] = static_cast<R>(.5);
    ttl::fill(ref(g_ls), static_cast<float>(1.0 / batch_size));

    const int n_epochs = 1;
    int step = 0;
    for (auto _ [[gnu::unused]] : ttl::range(n_epochs)) {
        for (auto offset : ttl::range(n / batch_size)) {
            ++step;
            printf("step: %d\n", step);
            // if (step > 2) { break; }

            load_data(ds, offset * batch_size, batch_size, ref(xs), ref(y_s));
            {
                slp()(ref(zs), view(xs), view(w), view(b));
                ttl::nn::ops::softmax()(ref(ys), view(zs));
                ttl::nn::ops::xentropy()(ref(ls), view(y_s), view(ys));
                ttl::nn::ops::mean()(ref(l), view(ls));
            }
            printf("loss: %f\n", l.data()[0]);
            {
                // PPRINT(g_ls);
                ttl::nn::ops::grad::xentropy<1>()(
                    ref(g_ys), view(g_ls), view(ls), view(y_s), view(ys));
                // PPRINT(g_ys);
                ttl::nn::ops::grad::softmax<0>()(ref(g_zs), view(g_ys),
                                                 view(ys), view(zs));
                // PPRINT(g_zs);
                slp().grad(ref(g_w), ref(g_b), view(g_zs), view(zs), view(xs),
                           view(w), view(b));
            }
            // PPRINT(g_w);
            // PPRINT(g_b);
            ttl::nn::ops::sub()(w, view(w), view(g_w));
            ttl::nn::ops::sub()(b, view(b), view(g_b));
        }
    }
}

template <typename D, typename R>
void test_slp_model(const D &ds, const ttl::tensor_view<R, 2> &w,
                    const ttl::tensor_view<R, 1> &b)
{
    const auto [n, height, width] = ds.images.shape().dims();
    const int k = 10;

    ttl::tensor<R, 2> xs(n, height * width);
    ttl::tensor<R, 2> ys(n, k);
    ttl::tensor<R, 2> y_s(n, k);
    load_data(ds, 0, n, ref(xs), ref(y_s));

    slp()(ref(ys), view(xs), w, b);
    const auto acc = accuracy<float>(view(ys), view(y_s));
    printf("accuracy: %f\n", acc);
}

int main()
{
    using nn::experimental::datasets::load_mnist_data;

    TRACE_SCOPE(__func__);

    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    const int k = 10;
    {
        TRACE_SCOPE("train");
        ttl::tensor<float, 2> w(28 * 28, k);
        ttl::tensor<float, 1> b(k);

        ttl::fill(ref(w), static_cast<float>(.5));
        ttl::fill(ref(b), static_cast<float>(0));

        train_slp_model(train, ref(w), ref(b));
        test_slp_model(test, view(w), view(b));

        ttl::nn::ops::writefile("w.idx")(view(w));
        ttl::nn::ops::writefile("b.idx")(view(b));
        test_slp_model(test, view(w), view(b));
    }
    int code [[gnu::unused]] = system("tar -cf params.idx.tar w.idx b.idx");
    {
        TRACE_SCOPE("test");
        ttl::tensor<float, 2> w(28 * 28, k);
        ttl::tensor<float, 1> b(k);

        fill(ref(w), static_cast<float>(0));
        fill(ref(b), static_cast<float>(0));

        const std::string filename("params.idx.tar");
        ttl::nn::ops::readtar(filename, "w.idx")(ref(w));
        ttl::nn::ops::readtar(filename, "b.idx")(ref(b));

        test_slp_model(test, view(w), view(b));
    }
    return 0;
}
