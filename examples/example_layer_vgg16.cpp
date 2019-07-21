// http://www.cs.toronto.edu/~frossard/post/vgg16/

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>
#include <vector>

#include <nn/layers>

#ifdef USE_OPENCV
#    include <opencv2/opencv.hpp>
#endif

#include "utils.hpp"

using ttl::range;

const size_t h = 224;
const size_t w = 224;
const size_t k = 1000;
const int m = 5;

template <typename R>
auto example_vgg16(const ttl::tensor_ref<R, 4> &x, const std::string &prefix)
{
    using image_order = nn::ops::nhwc;
    using filter_order = nn::ops::rscd;
    using relu = nn::ops::pointwise<nn::ops::relu>;
    using conv = nn::layers::conv<image_order, filter_order, true, relu>;
    using pool = nn::layers::pool<nn::ops::pool_max, image_order>;
    using dense_relu = nn::layers::dense<relu>;
    using dense = nn::layers::dense<>;
    using softmax = nn::layers::activation<nn::ops::softmax>;
    using top = nn::ops::top;

    const auto p = [prefix](auto name) {
        return nn::ops::readtar(prefix, name);
    };

    auto l1_1 = conv(64, conv::ksize(3, 3),
                     conv::padding_same())(x, p("conv1_1_W"), p("conv1_1_b"));
    auto l1_2 = conv(64, conv::ksize(3, 3), conv::padding_same())(
        ref(*l1_1), p("conv1_2_W"), p("conv1_2_b"));
    auto l1_3 = pool(pool::ksize(2, 2))(ref(*l1_2));
    PPRINT(*l1_3);

    auto l2_1 = conv(128, conv::ksize(3, 3), conv::padding_same())(
        ref(*l1_3), p("conv2_1_W"), p("conv2_1_b"));
    auto l2_2 = conv(128, conv::ksize(3, 3), conv::padding_same())(
        ref(*l2_1), p("conv2_2_W"), p("conv2_2_b"));
    auto l2_3 = pool(pool::ksize(2, 2))(ref(*l2_2));
    PPRINT(*l2_3);

    auto l3_1 = conv(256, conv::ksize(3, 3), conv::padding_same())(
        ref(*l2_3), p("conv3_1_W"), p("conv3_1_b"));
    auto l3_2 = conv(256, conv::ksize(3, 3), conv::padding_same())(
        ref(*l3_1), p("conv3_2_W"), p("conv3_2_b"));
    auto l3_3 = conv(256, conv::ksize(3, 3), conv::padding_same())(
        ref(*l3_2), p("conv3_3_W"), p("conv3_3_b"));
    auto l3_4 = pool(pool::ksize(2, 2))(ref(*l3_3));
    PPRINT(*l3_4);

    auto l4_1 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ref(*l3_4), p("conv4_1_W"), p("conv4_1_b"));
    auto l4_2 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ref(*l4_1), p("conv4_2_W"), p("conv4_2_b"));
    auto l4_3 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ref(*l4_2), p("conv4_3_W"), p("conv4_3_b"));
    auto l4_4 = pool(pool::ksize(2, 2))(ref(*l4_3));
    PPRINT(*l4_4);

    auto l5_1 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ref(*l4_4), p("conv5_1_W"), p("conv5_1_b"));
    auto l5_2 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ref(*l5_1), p("conv5_2_W"), p("conv5_2_b"));
    auto l5_3 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ref(*l5_2), p("conv5_3_W"), p("conv5_3_b"));
    auto l5_4 = pool(pool::ksize(2, 2))(ref(*l5_3));
    PPRINT(*l5_4);

    auto l5_flat = nn::ops::as_matrix<1, 3>(ref(*l5_4));
    auto l6 = dense_relu(4096)(l5_flat, p("fc6_W"), p("fc6_b"));
    auto l7 = dense_relu(4096)(ref(*l6), p("fc7_W"), p("fc7_b"));
    auto l8 = dense(k)(ref(*l7), p("fc8_W"), p("fc8_b"));
    auto out = softmax()(ref(*l8));
    PPRINT(*out);

    ttl::tensor<R, 1> y(m);
    ttl::tensor<nn::shape<1>::dimension_type, 1> z(m);
    (top(m))(ref(y), ref(z), view(*out)[0]);

    return std::make_pair(std::move(y), std::move(z));
}

template <typename T> typename T::value_type *data_end(const T &t)
{
    return t.data() + t.shape().size();
}

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
    auto x = ttl::tensor<float, 4>(1, h, w, 3);
    {
#ifdef USE_OPENCV
        system("[ ! -f laska.png ] && curl -vLOJ "
               "https://www.cs.toronto.edu/~frossard/vgg16/laska.png");
        auto img = cv::imread("laska.png");
        auto input = ttl::tensor<uint8_t, 4>(x.shape());
        cv::Mat resized_image(cv::Size(w, h), CV_8UC(3), input.data());
        cv::resize(img, resized_image, resized_image.size(), 0, 0);
        for (auto i : range(h)) {
            for (auto j : range(w)) {
                x.at(0, i, j, 0) = input.at(0, i, j, 2);
                x.at(0, i, j, 1) = input.at(0, i, j, 1);
                x.at(0, i, j, 2) = input.at(0, i, j, 0);
            }
        }
#else
        (nn::ops::readfile(prefix + "/laska.idx"))(ref(x)[0]);
#endif
        std::vector<float> mean({123.68, 116.779, 103.939});
        nn::ops::apply_bias<nn::ops::nhwc, std::minus<float>>()(
            ref(x), view(x), ttl::tensor_view<float, 1>(mean.data(), 3));
    }

    const auto [y, z] =
        example_vgg16(ref(x), prefix + "/vgg16_weights.idx.tar");

    for (auto i : range(m)) {
        printf("%u: %f %s\n", z.at(i), y.at(i), names[z.at(i)].c_str());
    }

    return 0;
}
