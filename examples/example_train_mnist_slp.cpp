#include <cstdlib>

#include <nn/experimental/bits/ops/utility.hpp>
#include <nn/experimental/datasets>
#include <nn/layers>
#include <nn/ops>

#include "trace.hpp"
#include "utils.hpp"

#include <experimental/range>

using std::experimental::range;

template <typename R>
void grad_softmax(const ttl::tensor_ref<R, 1> &gx,
                  const ttl::tensor_view<R, 1> &gy,
                  const ttl::tensor_view<R, 1> &y,
                  const ttl::tensor_view<R, 1> &x)
{
    const auto n = x.shape().size();
    const ttl::tensor<R, 2> g(n, n);
    for (auto i : range(n)) {
        for (auto j : range(n)) {
            if (i == j) {
                const R v = y.at(i);
                g.at(i, j) = v * (static_cast<R>(1) - v);
            } else {
                g.at(i, j) = -y.at(i) * y.at(j);
            }
        }
    }
    nn::engines::linag<nn::engines::default_engine>::vm(gy, view(g), gx);
}

template <typename R>
void grad_softmax(const ttl::tensor_ref<R, 2> &gx,
                  const ttl::tensor_view<R, 2> &gy,
                  const ttl::tensor_view<R, 2> &y,
                  const ttl::tensor_view<R, 2> &x)
{
    for (auto i : range(x.shape().dims[0])) {
        grad_softmax(gx[i], gy[i], y[i], x[i]);
    }
}

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
        using add_bias = nn::ops::apply_bias<nn::ops::hw, std::plus<R>>;
        nn::ops::matmul()(ys, xs, w);
        add_bias()(ref(ys), view(ys), b);
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
void grad_xentropy_y(const ttl::tensor_ref<R, 1> &gx,
                     const ttl::tensor_view<R, 0> &gz,
                     const ttl::tensor_view<R, 0> &z,
                     const ttl::tensor_view<R, 1> &x,
                     const ttl::tensor_view<R, 1> &y)
{
    for (auto i : range(y.shape().size())) {
        gx.data()[i] = gz.data()[0] * (-x.data()[i] / y.data()[i]);
    }
}

template <typename R>
void grad_xentropy_y(const ttl::tensor_ref<R, 2> &gx,
                     const ttl::tensor_view<R, 1> &gz,
                     const ttl::tensor_view<R, 1> &z,
                     const ttl::tensor_view<R, 2> &x,
                     const ttl::tensor_view<R, 2> &y)
{
    for (auto i : range(y.shape().dims[0])) {
        grad_xentropy_y(gx[i], gz[i], z[i], x[i], y[i]);
    }
}

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
    using nn::experimental::ops::argmax;
    const auto [n, k] = ys.shape().dims;
    int t = 0;
    int f = 0;
    for (auto l : range(n)) {
        const bool b = argmax(ys[l]) == argmax(y_s[l]);
        b ? ++t : ++f;
    }
    return static_cast<R>(t) / static_cast<R>(t + f);
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
    using nn::experimental::ops::fill;

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
    fill(ref(g_ls), static_cast<float>(1.0 / batch_size));

    const int n_epochs = 1;
    int step = 0;
    for (auto _ : range(n_epochs)) {
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
                grad_xentropy_y(ref(g_ys), view(g_ls), view(ls), view(y_s),
                                view(ys));
                // PPRINT(g_ys);
                grad_softmax(ref(g_zs), view(g_ys), view(ys), view(zs));
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

    ttl::tensor<float, 2> xs(n, height * width);
    ttl::tensor<float, 2> ys(n, k);
    ttl::tensor<float, 2> y_s(n, k);
    load_data(ds, 0, n, ref(xs), ref(y_s));

    slp()(ref(ys), view(xs), view(w), view(b));
    const float acc = accuracy(view(ys), view(y_s));
    printf("accuracy: %f\n", acc);
}

int main()
{
    using nn::experimental::datasets::load_mnist_data;
    using nn::experimental::ops::fill;

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

        fill(ref(w), static_cast<float>(.5));
        fill(ref(b), static_cast<float>(0));

        train_slp_model(train, ref(w), ref(b));
        test_slp_model(test, view(w), view(b));

        nn::ops::writefile("w.idx")(view(w));
        nn::ops::writefile("b.idx")(view(b));
        test_slp_model(test, view(w), view(b));
    }
    system("tar -cf params.idx.tar w.idx b.idx");
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
