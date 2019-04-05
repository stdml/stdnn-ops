#include <nn/layers>

#include "testing.hpp"
#include "testing_traits.hpp"

struct conv2d_layer_params_t {
    using dim_t = uint32_t;

    dim_t ksize_h;
    dim_t ksize_w;
    dim_t stride_h;
    dim_t stride_w;
    dim_t rate_h;
    dim_t rate_w;
    dim_t pad_h_left;
    dim_t pad_h_right;
    dim_t pad_w_left;
    dim_t pad_w_right;

    dim_t depth;
};

template <class image_order, class filter_order>
void test_conv2d_layer(conv2d_layer_params_t p)
{
    using conv2d = nn::layers::conv<image_order, filter_order>;
    {
        conv2d layer(conv2d::ksize(3, 3), p.depth);
    }
    {
        conv2d l1(ksize<conv2d>(p), p.depth);
        conv2d l2(ksize<conv2d>(p), p.depth, padding<conv2d>(p));
    }
}

void test_conv2d_layer_all()
{
    conv2d_layer_params_t p;
    test_conv2d_layer<nn::ops::nhwc, nn::ops::rscd>(p);
    test_conv2d_layer<nn::ops::nchw, nn::ops::dcrs>(p);
}
