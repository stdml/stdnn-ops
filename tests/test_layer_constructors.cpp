#include "testing_traits.hpp"

#include <ttl/nn/layers>
#include <ttl/nn/testing>

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
        conv2d layer(p.depth, conv2d::ksize(3, 3));
    }
    {
        conv2d l1(p.depth, ksize<conv2d>(p));
        conv2d l2(p.depth, ksize<conv2d>(p), conv2d::padding_same());
        // FIXME: support fixed padding 2d
        // conv2d l3(p.depth, ksize<conv2d>(p), stride<conv2d>(p));
        conv2d l4(p.depth, ksize<conv2d>(p), conv2d::padding_same(),
                  stride<conv2d>(p));
        conv2d l5(p.depth, ksize<conv2d>(p), conv2d::padding_same(),
                  stride<conv2d>(p), rate<conv2d>(p));
    }
}

void test_conv2d_layer_all()
{
    conv2d_layer_params_t p;
    test_conv2d_layer<nn::ops::nhwc, nn::ops::rscd>(p);
    test_conv2d_layer<nn::ops::nchw, nn::ops::dcrs>(p);
}

struct pool2d_layer_params_t {
    using dim_t = uint32_t;

    dim_t ksize_h;
    dim_t ksize_w;
    dim_t stride_h;
    dim_t stride_w;
    dim_t pad_h_left;
    dim_t pad_h_right;
    dim_t pad_w_left;
    dim_t pad_w_right;
};

template <typename pool_algo, typename image_order>
void test_conv2d_layer(const pool2d_layer_params_t &p)
{
    using pool2d = nn::layers::pool<pool_algo, image_order>;
    {
        pool2d l1;
        pool2d l2(pool2d::ksize(2, 2));
    }
    {
        pool2d l1(ksize<pool2d>(p));
        pool2d l2(ksize<pool2d>(p), pool2d::padding_same());
        pool2d l3(ksize<pool2d>(p), pool2d::padding_same(), stride<pool2d>(p));
        // FIXME: support fixed padding 2d
    }
}

template <class pool_method> void _test_pool2d_all_bind_1()
{
    pool2d_layer_params_t p;
    test_conv2d_layer<pool_method, nn::ops::nhwc>(p);
    test_conv2d_layer<pool_method, nn::ops::nchw>(p);
}

void test_pool2d_layer_all()
{
    _test_pool2d_all_bind_1<nn::ops::pool_max>();
    _test_pool2d_all_bind_1<nn::ops::pool_mean>();
}
