// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <nn/models>

#include "utils.hpp"

using nn::layers::with_init;
using nn::layers::debug::show_name;

template <typename image_order = nn::ops::nhwc,
          typename filter_order = nn::ops::rscd>
class plain50_model
{
    const size_t logits = 1000;

    using relu = nn::ops::pointwise<nn::ops::relu>;
    using bn_layer = nn::layers::batch_norm<image_order, relu>;

    using flatten = nn::layers::flatten<1, 3>;
    using conv = nn::layers::conv<image_order, filter_order, true>;
    using softmax = nn::layers::activation<nn::ops::softmax>;

    auto conv7x7(int d) const
    {
        using conv_relu =
            nn::layers::conv<image_order, filter_order, true, relu>;
        return with_init(conv_relu(d, conv_relu::ksize(7, 7),
                                   conv_relu::padding_same(),
                                   conv_relu::stride(2, 2)),
                         show_name("conv1/kernel"), show_name("conv1/bias"));
    }

    auto bn() const
    {
        return with_init(bn_layer(),  //
                                      //  show_name("mean"), show_name("var"),
                         nn::ops::noop(),  //
                         nn::ops::noop(),  //
                         show_name("bn_conv?/beta"),
                         show_name("bn_conv?/gamma"));
    }

    auto pool1() const
    {
        using pool_max = nn::layers::pool<nn::ops::pool_max, image_order>;
        return pool_max(pool_max::ksize(3, 3), pool_max::padding_same(),
                        pool_max::stride(2, 2));
    }

    auto pool2() const
    {
        using pool_mean = nn::layers::pool<nn::ops::pool_mean, image_order>;
        return pool_mean(pool_mean::ksize(7, 7));
    }

    auto conv1x1(int d, int s) const
    {
        return with_init(conv(d, conv::ksize(1, 1), conv::padding_same(),
                              conv::stride(s, s)),
                         show_name("res??_branch??/kernel"),
                         show_name("res??_branch??/bias"));
    }

    auto conv3x3(int d, int s) const
    {
        return with_init(conv(d, conv::ksize(3, 3), conv::padding_same(),
                              conv::stride(s, s)),
                         show_name("res??_branch??/kernel"),
                         show_name("res??_branch??/bias"));
    }

    // auto conv2_x() const {}
    // auto conv3_x() const {}
    // auto conv4_x() const {}
    // auto conv5_x() const {}

    auto dense(int logits) const
    {
        using layer = nn::layers::dense<>;
        return with_init(layer(logits), show_name("W"), show_name("b"));
    }

    const std::string prefix_;

    const auto p(const std::string &name) const
    {
        return nn::ops::readtar(prefix_, name);
    }

  public:
    const size_t h = 224;
    const size_t w = 224;

    plain50_model(const std::string &prefix) : prefix_(prefix) {}

    template <typename R>
    auto operator()(const ttl::tensor_ref<R, 4> &x, int m = 5) const
    {
        auto layers = nn::models::make_sequential()  //
                      << conv7x7(64) << bn()         //
                      << pool1()                     //

                      // conv2_x [1x1, 64; 3x3, 64; 1x1, 256] x 3
                      << conv1x1(64, 1) << bn()   //
                      << conv3x3(64, 1) << bn()   //
                      << conv1x1(256, 1) << bn()  //

                      << conv1x1(64, 1) << bn()   //
                      << conv3x3(64, 1) << bn()   //
                      << conv1x1(256, 1) << bn()  //

                      << conv1x1(64, 1) << bn()  //
                      << conv3x3(64, 1) << bn()  //
                      << conv1x1(256, 1)
                      << bn()  //

                      // conv3_x [1x1, 128; 3x3, 128; 1x1, 512] x 4
                      << conv1x1(128, 2) << bn()  //
                      << conv3x3(128, 1) << bn()  //
                      << conv1x1(512, 1) << bn()  //

                      << conv1x1(128, 1) << bn()  //
                      << conv3x3(128, 1) << bn()  //
                      << conv1x1(512, 1) << bn()  //

                      << conv1x1(128, 1) << bn()  //
                      << conv3x3(128, 1) << bn()  //
                      << conv1x1(512, 1) << bn()  //

                      << conv1x1(128, 1) << bn()  //
                      << conv3x3(128, 1) << bn()  //
                      << conv1x1(512, 1)
                      << bn()  //

                      // conv4_x [1x1, 256; 3x3, 256; 1x1, 1024] x 6
                      << conv1x1(256, 2) << bn()   //
                      << conv3x3(256, 1) << bn()   //
                      << conv1x1(1024, 1) << bn()  //

                      << conv1x1(256, 1) << bn()   //
                      << conv3x3(256, 1) << bn()   //
                      << conv1x1(1024, 1) << bn()  //

                      << conv1x1(256, 1) << bn()   //
                      << conv3x3(256, 1) << bn()   //
                      << conv1x1(1024, 1) << bn()  //

                      << conv1x1(256, 1) << bn()   //
                      << conv3x3(256, 1) << bn()   //
                      << conv1x1(1024, 1) << bn()  //

                      << conv1x1(256, 1) << bn()   //
                      << conv3x3(256, 1) << bn()   //
                      << conv1x1(1024, 1) << bn()  //

                      << conv1x1(256, 1) << bn()  //
                      << conv3x3(256, 1) << bn()  //
                      << conv1x1(1024, 1)
                      << bn()  //

                      // conv5_x [1x1, 512; 3x3, 512; 1x1, 2048] x 3
                      << conv1x1(512, 2) << bn()   //
                      << conv3x3(512, 1) << bn()   //
                      << conv1x1(2048, 1) << bn()  //

                      << conv1x1(512, 1) << bn()   //
                      << conv3x3(512, 1) << bn()   //
                      << conv1x1(2048, 1) << bn()  //

                      << conv1x1(512, 1) << bn()  //
                      << conv3x3(512, 1) << bn()  //
                      << conv1x1(2048, 1)
                      << bn()  //

                      //
                      << pool2()        //
                      << flatten()      //
                      << dense(logits)  //
                      << softmax()      //
            ;

        auto y = layers(x);
        return y;
    }
};

int main(int argc, char *argv[])
{
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/models/resnet";
    plain50_model model(prefix);
    const auto x = ttl::tensor<float, 4>(1, model.h, model.w, 3);
    const auto y = model(ref(x));
    PPRINT(x);
    PPRINT(*y);
    return 0;
}
