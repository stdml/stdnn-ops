// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <nn/models>

#include "utils.hpp"

using nn::ops::pad;

template <typename image_order = nn::ops::nhwc,
          typename filter_order = nn::ops::rscd>
class plain34_model
{
    const size_t logits = 1000;

    using relu = nn::ops::pointwise<nn::ops::relu>;
    using bn_layer = nn::layers::batch_norm<image_order, relu>;

    using flatten = nn::layers::flatten<1, 3>;
    using dense = nn::layers::dense<>;
    using softmax = nn::layers::activation<nn::ops::softmax>;

    auto conv1(int d) const
    {
        using layer = nn::layers::conv<image_order, filter_order, true, relu>;
        return layer(d, layer::ksize(7, 7), layer::padding_same(),
                     layer::stride(2, 2));
    }

    auto pool1() const
    {
        using layer = nn::layers::pool<nn::ops::pool_max, image_order>;
        return layer(layer::ksize(3, 3), layer::padding_same(),
                     layer::stride(2, 2));
    }

    auto pool2() const
    {
        using layer = nn::layers::pool<nn::ops::pool_mean, image_order>;
        return layer(layer::ksize(7, 7));
    }

    auto conv(int d, int s) const
    {
        using layer = nn::layers::conv<image_order, filter_order, false>;
        return layer(d, layer::ksize(3, 3), layer::padding_same(),
                     layer::stride(s, s));
    }

    const std::string prefix_;

    const auto p(const std::string &name) const
    {
        return nn::ops::readtar(prefix_, name);
    }

  public:
    const size_t h = 224;
    const size_t w = 224;

    plain34_model(const std::string &prefix) : prefix_(prefix) {}

    template <typename R>
    auto operator()(const ttl::tensor_ref<R, 4> &x, int m = 5) const
    {
        auto layers = nn::models::make_sequential()  //
                      << conv1(64)                   //
                      << pool1()                     //

                      << conv(64, 1) << bn_layer()  //
                      << conv(64, 1) << bn_layer() << conv(64, 1) << bn_layer()
                      << conv(64, 1) << bn_layer() << conv(64, 1) << bn_layer()
                      << conv(64, 1) << bn_layer()

                      << conv(128, 2) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //
                      << conv(128, 1) << bn_layer()  //

                      << conv(256, 2) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //
                      << conv(256, 1) << bn_layer()  //

                      << conv(512, 2) << bn_layer()  //
                      << conv(512, 1) << bn_layer()  //
                      << conv(512, 1) << bn_layer()  //
                      << conv(512, 1) << bn_layer()  //
                      << conv(512, 1) << bn_layer()  //
                      << conv(512, 1) << bn_layer()  //

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
    plain34_model model(prefix);
    const auto x = ttl::tensor<float, 4>(1, model.h, model.w, 3);
    const auto y = model(ref(x));
    PPRINT(x);
    PPRINT(*y);
    return 0;
}
