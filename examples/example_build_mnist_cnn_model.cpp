#include <cstdlib>

#include <experimental/range>

#include <ttl/algorithm>

#include <nn/experimental/bits/ops/grad/softmax.hpp>
#include <nn/experimental/bits/ops/grad/xentropy.hpp>
#include <nn/experimental/bits/ops/utility.hpp>
#include <nn/experimental/datasets>
#include <nn/layers>
#include <nn/ops>

#include "model_builder.hpp"
#include "trace.hpp"
#include "utils.hpp"

using std::experimental::range;

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
        std::transform(pixels.data(), pixels.data_end(), xs[l].data(),
                       [](uint8_t p) { return p / 255.0; });
    }
}

template <typename R>
auto build_cnn_model(model_builder &b, const nn::shape<4> &input_shape,
                     int logits)
{
    TRACE_SCOPE("build_cnn_model");
    const auto [batch_size, height, width, channel] = input_shape.dims();

    const auto xs = b.var<R>(input_shape);
    const auto y_s = b.var<R>(batch_size, logits);

    const int d = 64;
    const auto w1 = b.covar<R>(nn::shape<4>(5, 5, 1, d));
    const auto b1 = b.covar<R>(nn::shape<1>(d));

    const auto l1 = b.apply<R>(nn::ops::add_bias<nn::ops::nhwc>(),
                               b.apply<R>(nn::ops::conv<>(), xs, w1), b1);
    const auto l1_flat =
        b.reshape(l1, nn::ops::as_mat_shape<1, 3>(l1.output().shape()));

    const auto w2 =
        b.covar<R>(nn::shape<2>(l1_flat.output().shape().dims()[1], logits));
    const auto b2 = b.covar<R>(nn::shape<1>(logits));
    const auto l2 = b.apply<R>(nn::ops::add_bias<nn::ops::hw>(),
                               b.apply<R>(nn::ops::matmul(), l1_flat, w2), b2);

    const auto probs = b.apply<R>(nn::ops::softmax(), l2);
    const auto loss = b.apply<R>(nn::ops::xentropy(), y_s, probs);

    const auto predictions =
        b.apply<uint32_t>(nn::experimental::ops::argmax(), probs);
    const auto labels = b.apply<uint32_t>(nn::experimental::ops::argmax(), y_s);

    const auto accuracy = b.apply<float>(nn::experimental::ops::similarity(),
                                         predictions, labels);
    return std::make_tuple(xs, y_s, w1, b1, w2, b2, loss, accuracy);
}

int main()
{
    using nn::experimental::datasets::load_mnist_data;

    TRACE_SCOPE(__func__);

    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    {
        TRACE_SCOPE("test");
        const int batch_size = 10000;
        const int height = 28;
        const int width = 28;
        model_builder b;
        const auto [xs, y_s, w1, b1, w2, b2, loss, accuracy] =
            build_cnn_model<float>(
                b, nn::shape<4>(batch_size, height, width, 1), 10);

        printf("w1: %s\n", show_shape(w1.output().shape()).c_str());
        printf("b1: %s\n", show_shape(b1.output().shape()).c_str());
        printf("w2: %s\n", show_shape(w2.output().shape()).c_str());
        printf("b2: %s\n", show_shape(b2.output().shape()).c_str());

        printf("loss: %s\n", show_shape(loss.output().shape()).c_str());
        printf("accuracy: %s\n", show_shape(accuracy.output().shape()).c_str());

        // const std::string filename("params.idx.tar");
        // nn::ops::readtar(filename, "w.idx")(ref(w1.output()));
        // nn::ops::readtar(filename, "b.idx")(ref(b1.output()));

        // load_data(test, 0, batch_size, ref(xs.output()), ref(y_s.output()));
        // accuracy();
        // printf("accuracy: %f\n", accuracy.output().data()[0]);
    }
    return 0;
}
