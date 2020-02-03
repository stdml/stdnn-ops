// http://www.cs.toronto.edu/~frossard/post/vgg16/

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <ttl/nn/models>

#ifdef USE_OPENCV
#    include <opencv2/opencv.hpp>
#endif

#include "utils.hpp"

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
    auto operator()(const ttl::tensor_ref<R, 4> &x, int m) const
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
        auto l5_flat = ttl::nn::ops::as_matrix<1, 3>(ref(*l5_4));
        PPRINT(l5_flat);
        auto out = dense_layers(l5_flat);
        PPRINT(*out);

        ttl::tensor<R, 1> y(m);
        ttl::tensor<ttl::shape<1>::dimension_type, 1> z(m);
        (top(m))(ref(y), ref(z), view(*out)[0]);

        return std::make_pair(std::move(y), std::move(z));
    }
};

std::vector<std::string> load_class_names(const std::string &filename)
{
    std::vector<std::string> names;
    std::string line;
    std::ifstream in(filename);
    while (std::getline(in, line)) { names.push_back(line); }
    return names;
}

int main(int argc, char *argv[])
{
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/models/vgg16";
    const auto names = load_class_names(prefix + "/vgg16-class-names.txt");
    vgg16_model vgg16(prefix + "/vgg16_weights.idx.tar");
    auto x = ttl::tensor<float, 4>(1, vgg16.h, vgg16.w, 3);
    {
#ifdef USE_OPENCV
        system("[ ! -f laska.png ] && curl -vLOJ "
               "https://www.cs.toronto.edu/~frossard/vgg16/laska.png");
        auto img = cv::imread("laska.png");
        auto input = ttl::tensor<uint8_t, 4>(x.shape());
        cv::Mat resized_image(cv::Size(vgg16.w, vgg16.h), CV_8UC(3),
                              input.data());
        cv::resize(img, resized_image, resized_image.size(), 0, 0);
        for (auto i : ttl::range(vgg16.h)) {
            for (auto j : ttl::range(vgg16.w)) {
                x.at(0, i, j, 0) = input.at(0, i, j, 2);
                x.at(0, i, j, 1) = input.at(0, i, j, 1);
                x.at(0, i, j, 2) = input.at(0, i, j, 0);
            }
        }
#else
        (ttl::nn::ops::readfile(prefix + "/laska.idx"))(ref(x)[0]);
#endif
        std::vector<float> mean({123.68, 116.779, 103.939});
        ttl::nn::ops::apply_bias<ttl::nn::ops::nhwc, std::minus<float>>()(
            ref(x), view(x),
            ttl::tensor_view<float, 1>(mean.data(), mean.size()));
    }

    int m = 5;
    const auto [y, z] = vgg16(ref(x), m);

    for (auto i : ttl::range(m)) {
        printf("%u: %f %s\n", z.at(i), y.at(i), names[z.at(i)].c_str());
    }

    return 0;
}
