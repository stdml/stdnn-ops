#include <ttl/nn/testing>
#include <ttl/nn/traits>

TEST(traits_test, test_1)
{
    using dim_t = ttl::shape<3>::dimension_type;
    ttl::shape<3> s(4, 5, 6);

    ASSERT_EQ(ttl::nn::ops::channel_size<ttl::nn::ops::chw>(s), (dim_t)4);
    ASSERT_EQ(ttl::nn::ops::channel_size<ttl::nn::ops::hwc>(s), (dim_t)6);

    ASSERT_EQ(ttl::nn::ops::image_shape<ttl::nn::ops::chw>(s),
              ttl::shape<2>(5, 6));
    ASSERT_EQ(ttl::nn::ops::image_shape<ttl::nn::ops::hwc>(s),
              ttl::shape<2>(4, 5));
}

TEST(traits_test, test_2)
{
    using dim_t = ttl::shape<3>::dimension_type;
    ttl::shape<4> s(3, 4, 5, 6);

    ASSERT_EQ(ttl::nn::ops::channel_size<ttl::nn::ops::nchw>(s), (dim_t)4);
    ASSERT_EQ(ttl::nn::ops::channel_size<ttl::nn::ops::nhwc>(s), (dim_t)6);

    ASSERT_EQ(ttl::nn::ops::image_shape<ttl::nn::ops::nchw>(s),
              ttl::shape<2>(5, 6));
    ASSERT_EQ(ttl::nn::ops::image_shape<ttl::nn::ops::nhwc>(s),
              ttl::shape<2>(4, 5));
}

TEST(traits_test, test_3)
{
    using dim_t = ttl::shape<3>::dimension_type;
    ttl::shape<4> s(3, 4, 5, 6);

    ASSERT_EQ(ttl::nn::ops::filter_shape<ttl::nn::ops::rscd>(s),
              ttl::shape<2>(3, 4));
    ASSERT_EQ(ttl::nn::ops::filter_in_channel_size<ttl::nn::ops::rscd>(s),
              (dim_t)5);
    ASSERT_EQ(ttl::nn::ops::filter_out_channel_size<ttl::nn::ops::rscd>(s),
              (dim_t)6);

    ASSERT_EQ(ttl::nn::ops::filter_shape<ttl::nn::ops::dcrs>(s),
              ttl::shape<2>(5, 6));
    ASSERT_EQ(ttl::nn::ops::filter_in_channel_size<ttl::nn::ops::dcrs>(s),
              (dim_t)4);
    ASSERT_EQ(ttl::nn::ops::filter_out_channel_size<ttl::nn::ops::dcrs>(s),
              (dim_t)3);
}
