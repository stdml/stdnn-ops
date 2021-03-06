// http://www.cs.toronto.edu/~frossard/post/vgg16/

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <ttl/nn/models>

#include "utils.hpp"
#include "vgg16_common.hpp"

class vgg16_model
{
    const size_t k = 1000;

    using image_order = ttl::nn::ops::nhwc;
    using filter_order = ttl::nn::ops::rscd;

    using relu = ttl::nn::ops::relu;
    using pool = ttl::nn::layers::pool<ttl::nn::ops::pool_max, image_order>;
    using dense_relu = ttl::nn::layers::dense<relu>;
    using dense = ttl::nn::layers::dense<>;
    using softmax = ttl::nn::layers::activation<ttl::nn::ops::softmax>;
    using top = ttl::nn::ops::top;

    auto conv(int d) const
    {
        using conv_layer =
            ttl::nn::layers::conv<image_order, filter_order, true, relu>;
        return conv_layer(d, conv_layer::ksize(3, 3),
                          conv_layer::padding_same());
    }

    const std::string prefix_;

    const auto p(const std::string &name) const
    {
        return ttl::nn::ops::readtar(prefix_, name);
    }

  public:
    const size_t h = 224;
    const size_t w = 224;

    vgg16_model(const std::string &prefix) : prefix_(prefix) {}

    template <typename R>
    auto operator()(const ttl::tensor_view<R, 4> &x, int m) const
    {
        using ttl::nn::layers::with_init;

        auto conv_layers =
            ttl::nn::models::make_sequential()
            << with_init(conv(64), p("conv1_1_W"), p("conv1_1_b"))
            << with_init(conv(64), p("conv1_2_W"), p("conv1_2_b"))
            << pool(pool::ksize(2, 2))
            << with_init(conv(128), p("conv2_1_W"), p("conv2_1_b"))
            << with_init(conv(128), p("conv2_2_W"), p("conv2_2_b"))
            << pool(pool::ksize(2, 2))
            << with_init(conv(256), p("conv3_1_W"), p("conv3_1_b"))
            << with_init(conv(256), p("conv3_2_W"), p("conv3_2_b"))
            << with_init(conv(256), p("conv3_3_W"), p("conv3_3_b"))
            << pool(pool::ksize(2, 2))
            << with_init(conv(512), p("conv4_1_W"), p("conv4_1_b"))
            << with_init(conv(512), p("conv4_2_W"), p("conv4_2_b"))
            << with_init(conv(512), p("conv4_3_W"), p("conv4_3_b"))
            << pool(pool::ksize(2, 2))
            << with_init(conv(512), p("conv5_1_W"), p("conv5_1_b"))
            << with_init(conv(512), p("conv5_2_W"), p("conv5_2_b"))
            << with_init(conv(512), p("conv5_3_W"), p("conv5_3_b"))
            << pool(pool::ksize(2, 2));

        auto dense_layers =
            ttl::nn::models::make_sequential()
            << with_init(dense_relu(4096), p("fc6_W"), p("fc6_b"))
            << with_init(dense_relu(4096), p("fc7_W"), p("fc7_b"))
            << with_init(dense(k), p("fc8_W"), p("fc8_b")) << softmax();

        auto l5_4 = conv_layers(x);
        PPRINT(*l5_4);
        auto l5_flat = ttl::nn::ops::as_matrix<1, 3>(ttl::view(*l5_4));
        PPRINT(l5_flat);
        auto out = dense_layers(l5_flat);
        PPRINT(*out);

        ttl::tensor<R, 1> y(m);
        ttl::tensor<ttl::shape<1>::dimension_type, 1> z(m);
        (top(m))(ttl::ref(y), ttl::ref(z), view(*out)[0]);

        return std::make_pair(std::move(y), std::move(z));
    }
};

int main(int argc, char *argv[])
{
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/models/vgg16";
    const auto names = load_class_names(prefix + "/vgg16-class-names.txt");
    vgg16_model vgg16(prefix + "/vgg16_weights.idx.tar");
    auto x = ttl::tensor<float, 4>(1, vgg16.h, vgg16.w, 3);
    read_example_image(prefix, ttl::ref(x));
    const int m = 5;
    const auto [y, z] = vgg16(ttl::view(x), m);
    for (auto i : ttl::range(m)) {
        printf("%u: %f %s\n", z.at(i), y.at(i), names[z.at(i)].c_str());
    }
    return 0;
}
