#include <nn/ops>

#include "testing.hpp"
#include "testing_traits.hpp"

struct pool2d_params_t {
    using dim_t = uint32_t;

    dim_t ksize_h;
    dim_t ksize_w;
    dim_t stride_h;
    dim_t stride_w;
    dim_t pad_h_left;
    dim_t pad_h_right;
    dim_t pad_w_left;
    dim_t pad_w_right;

    using trait_t = nn::ops::pool_trait<nn::ops::hw>;
};

template <class pool2d> void test_pool2d_constructors(pool2d_params_t p)
{
    pool2d op1;
    pool2d op2(ksize<pool2d>(p));
    pool2d op3(ksize<pool2d>(p), padding<pool2d>(p));
    pool2d op4(ksize<pool2d>(p), stride<pool2d>(p));
    pool2d op5(ksize<pool2d>(p), padding<pool2d>(p), stride<pool2d>(p));
}

template <class pool_method, class image_order>
void test_pool2d(pool2d_params_t p)
{
    using pool2d = nn::ops::pool<pool_method, image_order>;
    static_assert(std::is_base_of<pool2d_params_t::trait_t, pool2d>::value);
    {
        pool2d op(
            pool2d::ksize(p.ksize_h, p.ksize_w),
            pool2d::padding(pool2d::padding_1d(p.pad_h_left, p.pad_h_right),
                            pool2d::padding_1d(p.pad_w_left, p.pad_w_right)),
            pool2d::stride(p.stride_h, p.stride_w));
    }
    {
        test_pool2d_constructors<pool2d>(p);
        test_pool2d_constructors<pool2d_params_t::trait_t>(p);
    }
    {
        pool2d_params_t::trait_t t;
        pool2d op(t);
    }
}

template <class pool_method> void _test_pool2d_all_bind_1()
{
    pool2d_params_t p;
    test_pool2d<pool_method, nn::ops::nhwc>(p);
    test_pool2d<pool_method, nn::ops::nchw>(p);
}

void test_pool2d_all()
{
    _test_pool2d_all_bind_1<nn::ops::pool_max>();
    _test_pool2d_all_bind_1<nn::ops::pool_mean>();
}

struct im2col_params_t {
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

    using trait_t = nn::ops::im2col_trait<nn::ops::hw>;
};

template <class image_order, class col_order>
void test_im2col(im2col_params_t p)
{
    using im2col = nn::ops::im2col<image_order, col_order>;
    static_assert(std::is_base_of<im2col_params_t::trait_t, im2col>::value);
    {
        im2col op(
            im2col::ksize(p.ksize_h, p.ksize_w),
            im2col::padding(im2col::padding_1d(p.pad_h_left, p.pad_h_right),
                            im2col::padding_1d(p.pad_w_left, p.pad_w_right)),
            im2col::stride(p.stride_h, p.stride_w),
            im2col::rate(p.rate_h, p.rate_w));
    }
    {
        im2col op(ksize<im2col>(p), padding<im2col>(p), stride<im2col>(p),
                  rate<im2col>(p));
    }
}

void test_im2col_all()
{
    im2col_params_t p;
    test_im2col<nn::ops::hw, nn::ops::rshw>(p);
    test_im2col<nn::ops::hw, nn::ops::hwrs>(p);
    test_im2col<nn::ops::hwc, nn::ops::hwrsc>(p);
}

struct conv2d_params_t {
    using dim_t = uint32_t;

    dim_t stride_h;
    dim_t stride_w;
    dim_t rate_h;
    dim_t rate_w;
    dim_t pad_h_left;
    dim_t pad_h_right;
    dim_t pad_w_left;
    dim_t pad_w_right;

    using trait_t = nn::ops::conv_trait<nn::ops::hw>;
};

template <class conv2d> void test_conv2d_constructors(conv2d_params_t p)
{
    conv2d op1;
    conv2d op2(padding<conv2d>(p));
    conv2d op3(stride<conv2d>(p));
    conv2d op4(padding<conv2d>(p), stride<conv2d>(p));
    conv2d op5(padding<conv2d>(p), stride<conv2d>(p), rate<conv2d>(p));
}

template <class image_order, class filter_order>
void test_conv2d(conv2d_params_t p)
{
    using conv2d = nn::ops::conv<image_order, filter_order>;
    static_assert(std::is_base_of<conv2d_params_t::trait_t, conv2d>::value);
    {
        conv2d op(
            conv2d::padding(conv2d::padding_1d(p.pad_h_left, p.pad_h_right),
                            conv2d::padding_1d(p.pad_w_left, p.pad_w_right)),
            conv2d::stride(p.stride_h, p.stride_w),
            conv2d::rate(p.rate_h, p.rate_w));
    }
    {
        test_conv2d_constructors<conv2d>(p);
        test_conv2d_constructors<conv2d_params_t::trait_t>(p);
    }
    {
        conv2d_params_t::trait_t t;
        conv2d op(t);
    }
}

void test_conv2d_all()
{
    conv2d_params_t p;
    test_conv2d<nn::ops::nhwc, nn::ops::rscd>(p);
    test_conv2d<nn::ops::nchw, nn::ops::dcrs>(p);
}
