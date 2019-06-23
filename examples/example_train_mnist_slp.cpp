#include <cstdlib>

#include <experimental/range>

#include <ttl/algorithm>

#include <nn/experimental/bits/ops/grad/softmax.hpp>
#include <nn/experimental/bits/ops/grad/xentropy.hpp>
#include <nn/experimental/bits/ops/utility.hpp>
#include <nn/experimental/datasets>
#include <nn/layers>
#include <nn/ops>

#include "trace.hpp"
#include "utils.hpp"

using std::experimental::range;

class slp
{
  public:
    nn::shape<2> operator()(const nn::shape<2> &x, const nn::shape<2> &y,
                            const nn::shape<1> &z) const
    {
        const auto [n, m] = x.dims;
        const auto [_m, k] = y.dims;
        contract_assert_eq(m, _m);
        contract_assert_eq(k, std::get<0>(z.dims));
        return nn::shape<2>(n, k);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &ys,
                    const ttl::tensor_view<R, 2> &xs,
                    const ttl::tensor_view<R, 2> &w,
                    const ttl::tensor_view<R, 1> &b)
    {
        nn::ops::matmul()(ys, xs, w);
        nn::ops::add_bias<nn::ops::hw>()(ref(ys), view(ys), b);
    }

    template <typename R>
    void
    grad(const ttl::tensor_ref<R, 2> &g_w, const ttl::tensor_ref<R, 1> &g_b,
         const ttl::tensor_view<R, 2> &g_ys, const ttl::tensor_view<R, 2> &ys,
         const ttl::tensor_view<R, 2> &xs, const ttl::tensor_view<R, 2> &w,
         const ttl::tensor_view<R, 1> &b)
    {
        const auto [n, k] = g_ys.shape().dims;
        for (auto l : range(k)) {
            R tot = 0;
            for (auto i : range(n)) { tot += g_ys.at(i, l); }
            g_b.at(l) = tot;
        }
        nn::engines::linag<nn::engines::default_engine>::mtm(xs, g_ys, g_w);
    }
};

template <typename R>
void loss(const ttl::tensor_ref<R, 0> &l, const ttl::tensor_view<R, 2> &ys,
          const ttl::tensor_view<R, 2> &y_s)
{
    ttl::tensor<R, 1> ls(ys.shape().dims[0]);
    nn::ops::xentropy()(ref(ls), y_s, ys);
    l.data()[0] = nn::ops::summaries::mean()(view(ls));
}

template <typename R>
R accuracy(const ttl::tensor_view<R, 2> &ys, const ttl::tensor_view<R, 2> &y_s)
{
    const nn::experimental::ops::argmax argmax;
    const ttl::tensor<uint32_t, 1> preditions(argmax(ys.shape()));
    const ttl::tensor<uint32_t, 1> labels(argmax(y_s.shape()));
    argmax(ref(preditions), ys);
    argmax(ref(labels), y_s);
    const auto diff = ttl::hamming_distance(view(preditions), view(labels));
    return 1 - static_cast<R>(diff) / static_cast<R>(labels.shape().size());
}

template <typename D, typename R>
void load_data(const D &ds, int offset, int batch_size,
               const ttl::tensor_ref<R, 2> &xs,
               const ttl::tensor_ref<R, 2> &y_s)
{
    using nn::experimental::ops::onehot;

    onehot(10)(y_s, view(ds.second.slice(offset, offset + batch_size)));
    const auto images = ds.first.slice(offset, offset + batch_size);
    for (auto l : range(batch_size)) {
        const auto pixels = images[l];
        std::transform(pixels.data(), pixels.data() + pixels.shape().size(),
                       xs[l].data(), [](uint8_t p) { return p / 255.0; });
    }
}

template <typename D, typename R>
void train_slp_model(const D &ds,  //
                     const ttl::tensor_ref<R, 2> &w,
                     const ttl::tensor_ref<R, 1> &b,  //
                     const int batch_size = 100)
{
    const auto [n, height, width] = ds.first.shape().dims;
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
    for (auto _ : range(n_epochs)) {
        UNUSED(_);
        for (auto offset : range(n / batch_size)) {
            ++step;
            printf("step: %d\n", step);
            // if (step > 2) { break; }

            load_data(ds, offset * batch_size, batch_size, ref(xs), ref(y_s));
            {
                slp()(ref(zs), view(xs), view(w), view(b));
                nn::ops::softmax()(ref(ys), view(zs));
                nn::ops::xentropy()(ref(ls), view(y_s), view(ys));
                l.data()[0] = nn::ops::summaries::mean()(view(ls));
            }
            printf("loss: %f\n", l.data()[0]);
            {
                // PPRINT(g_ls);
                nn::experimental::ops::grad::xentropy<1>()(
                    ref(g_ys), view(g_ls), view(ls), view(y_s), view(ys));
                // PPRINT(g_ys);
                nn::experimental::ops::grad::softmax<0>()(ref(g_zs), view(g_ys),
                                                          view(ys), view(zs));
                // PPRINT(g_zs);
                slp().grad(ref(g_w), ref(g_b), view(g_zs), view(zs), view(xs),
                           view(w), view(b));
            }
            // PPRINT(g_w);
            // PPRINT(g_b);
            nn::ops::sub()(w, view(w), view(g_w));
            nn::ops::sub()(b, view(b), view(g_b));
        }
    }
}

template <typename D, typename R>
void test_slp_model(const D &ds, const ttl::tensor_view<R, 2> &w,
                    const ttl::tensor_view<R, 1> &b)
{
    const auto [n, height, width] = ds.first.shape().dims;
    const int k = 10;

    ttl::tensor<R, 2> xs(n, height * width);
    ttl::tensor<R, 2> ys(n, k);
    ttl::tensor<R, 2> y_s(n, k);
    load_data(ds, 0, n, ref(xs), ref(y_s));

    slp()(ref(ys), view(xs), view(w), view(b));
    const R acc = accuracy(view(ys), view(y_s));
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

        nn::ops::writefile("w.idx")(view(w));
        nn::ops::writefile("b.idx")(view(b));
        test_slp_model(test, view(w), view(b));
    }
    int code = system("tar -cf params.idx.tar w.idx b.idx");
    UNUSED(code);
    {
        TRACE_SCOPE("test");
        ttl::tensor<float, 2> w(28 * 28, k);
        ttl::tensor<float, 1> b(k);

        fill(ref(w), static_cast<float>(0));
        fill(ref(b), static_cast<float>(0));

        const std::string filename("params.idx.tar");
        nn::ops::readtar(filename, "w.idx")(ref(w));
        nn::ops::readtar(filename, "b.idx")(ref(b));

        test_slp_model(test, view(w), view(b));
    }
    return 0;
}
