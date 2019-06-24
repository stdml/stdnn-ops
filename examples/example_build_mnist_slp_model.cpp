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
        std::transform(pixels.data(), pixels.data() + pixels.shape().size(),
                       xs[l].data(), [](uint8_t p) { return p / 255.0; });
    }
}

template <typename R, typename D>
void build_slp_model(const D &ds, const int batch_size = 100)
{
    const auto [n, height, width] = ds.first.shape().dims;
    const int k = 10;

    model_builder b;

    const auto xs = b.var<R>(batch_size, height * width);
    const auto y_s = b.var<R>(batch_size, k);

    const auto w1 = b.var<R>(height * width, k);
    const auto b1 = b.var<R>(k);

    const auto l1 = b.apply<R>(nn::ops::add_bias<nn::ops::hw>(),
                               b.apply<R>(nn::ops::matmul(), xs, w1), b1);

    const auto probs = b.apply<R>(nn::ops::softmax(), l1);
    const auto loss = b.apply<R>(nn::ops::xentropy(), y_s, probs);

    const auto predictions =
        b.apply<uint32_t>(nn::experimental::ops::argmax(), probs);
    const auto labels = b.apply<uint32_t>(nn::experimental::ops::argmax(), y_s);

    const auto acc = b.apply<float>(nn::experimental::ops::similarity(),
                                    predictions, labels);

    printf("loss: %s\n", show_shape(loss.output().shape()).c_str());
    printf("accuracy: %s\n", show_shape(acc.output().shape()).c_str());
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
        TRACE_SCOPE("train");
        build_slp_model<float>(train);
        build_slp_model<float>(test);
    }

    return 0;
}
