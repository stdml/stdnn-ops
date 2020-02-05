// http://www.cs.toronto.edu/~frossard/post/vgg16/

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>
#include <vector>

#include <ttl/nn/layers>

#include "utils.hpp"
#include "vgg16_common.hpp"

const size_t h = 224;
const size_t w = 224;
const size_t k = 1000;
const int m = 5;

template <typename R>
auto example_vgg16(const ttl::tensor_view<R, 4> &x, const std::string &prefix)
{
    using image_order = ttl::nn::ops::nhwc;
    using filter_order = ttl::nn::ops::rscd;
    using relu = ttl::nn::ops::relu;
    using conv = ttl::nn::layers::conv<image_order, filter_order, true, relu>;
    using pool = ttl::nn::layers::pool<ttl::nn::ops::pool_max, image_order>;
    using dense_relu = ttl::nn::layers::dense<relu>;
    using dense = ttl::nn::layers::dense<>;
    using softmax = ttl::nn::layers::activation<ttl::nn::ops::softmax>;
    using top = ttl::nn::ops::top;

    const auto p = [prefix](auto name) {
        return ttl::nn::ops::readtar(prefix, name);
    };

    auto l1_1 = conv(64, conv::ksize(3, 3),
                     conv::padding_same())(x, p("conv1_1_W"), p("conv1_1_b"));
    auto l1_2 = conv(64, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l1_1), p("conv1_2_W"), p("conv1_2_b"));
    auto l1_3 = pool(pool::ksize(2, 2))(ttl::view(*l1_2));
    PPRINT(*l1_3);

    auto l2_1 = conv(128, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l1_3), p("conv2_1_W"), p("conv2_1_b"));
    auto l2_2 = conv(128, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l2_1), p("conv2_2_W"), p("conv2_2_b"));
    auto l2_3 = pool(pool::ksize(2, 2))(ttl::view(*l2_2));
    PPRINT(*l2_3);

    auto l3_1 = conv(256, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l2_3), p("conv3_1_W"), p("conv3_1_b"));
    auto l3_2 = conv(256, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l3_1), p("conv3_2_W"), p("conv3_2_b"));
    auto l3_3 = conv(256, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l3_2), p("conv3_3_W"), p("conv3_3_b"));
    auto l3_4 = pool(pool::ksize(2, 2))(ttl::view(*l3_3));
    PPRINT(*l3_4);

    auto l4_1 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l3_4), p("conv4_1_W"), p("conv4_1_b"));
    auto l4_2 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l4_1), p("conv4_2_W"), p("conv4_2_b"));
    auto l4_3 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l4_2), p("conv4_3_W"), p("conv4_3_b"));
    auto l4_4 = pool(pool::ksize(2, 2))(ttl::view(*l4_3));
    PPRINT(*l4_4);

    auto l5_1 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l4_4), p("conv5_1_W"), p("conv5_1_b"));
    auto l5_2 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l5_1), p("conv5_2_W"), p("conv5_2_b"));
    auto l5_3 = conv(512, conv::ksize(3, 3), conv::padding_same())(
        ttl::view(*l5_2), p("conv5_3_W"), p("conv5_3_b"));
    auto l5_4 = pool(pool::ksize(2, 2))(ttl::view(*l5_3));
    PPRINT(*l5_4);

    auto l5_flat = ttl::nn::ops::as_matrix<1, 3>(ttl::view(*l5_4));
    auto l6 = dense_relu(4096)(l5_flat, p("fc6_W"), p("fc6_b"));
    auto l7 = dense_relu(4096)(ttl::view(*l6), p("fc7_W"), p("fc7_b"));
    auto l8 = dense(k)(ttl::view(*l7), p("fc8_W"), p("fc8_b"));
    auto out = softmax()(ttl::view(*l8));
    PPRINT(*out);

    ttl::tensor<R, 1> y(m);
    ttl::tensor<ttl::shape<1>::dimension_type, 1> z(m);
    (top(m))(ttl::ref(y), ttl::ref(z), view(*out)[0]);

    return std::make_pair(std::move(y), std::move(z));
}

int main(int argc, char *argv[])
{
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/models/vgg16";
    const auto names = load_class_names(prefix + "/vgg16-class-names.txt");
    auto x = ttl::tensor<float, 4>(1, h, w, 3);
    read_example_image(prefix, ttl::ref(x));
    const auto [y, z] =
        example_vgg16(ttl::view(x), prefix + "/vgg16_weights.idx.tar");
    for (auto i : ttl::range(m)) {
        printf("%u: %f %s\n", z.at(i), y.at(i), names[z.at(i)].c_str());
    }
    return 0;
}
