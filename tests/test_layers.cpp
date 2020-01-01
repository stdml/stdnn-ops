#include <ttl/nn/layers>
#include <ttl/nn/testing>

template <typename dense> void test_dense_layer()
{
    static_assert(std::is_class<dense>::value, "");

    auto x = ttl::tensor<float, 2>(2, 8);
    dense l1(10);
    l1(ref(x));
}

template <typename conv_layer> void test_conv_layer()
{
    static_assert(std::is_class<conv_layer>::value, "");
    // static_assert(std::is_constructible<L>::value, "");

    {
        auto x = ttl::tensor<float, 4>(2, 32, 32, 32);
        conv_layer l1(32, conv_layer::ksize(3, 3));
        l1(ref(x));
        conv_layer l2(32, conv_layer::ksize(3, 3), conv_layer::padding_same());
        l2(ref(x));
    }

    {
        auto x = ttl::tensor<float, 4>(2, 224, 224, 224);
        conv_layer l1(1, conv_layer::ksize(7, 7), conv_layer::padding_same(),
                      conv_layer::stride(2, 2));
        auto l = l1(ref(x));
        const auto shp = (*l).shape();
        ASSERT_EQ(
            shp.size(),
            static_cast<typename decltype(shp)::dimension_type>(2 * 112 * 112));
    }
}

template <typename pool> void test_pool_layer()
{
    static_assert(std::is_class<pool>::value, "");
    // static_assert(std::is_constructible<L>::value, "");

    auto x = ttl::tensor<float, 4>(2, 32, 32, 32);
    pool l1(pool::ksize(2, 2));
    l1(ref(x));
}

TEST(layers_test, test_1)
{
    test_dense_layer<ttl::nn::layers::dense<>>();
    test_dense_layer<
        ttl::nn::layers::dense<ttl::nn::ops::pointwise<ttl::nn::ops::relu>>>();

    test_conv_layer<ttl::nn::layers::conv<>>();
    test_conv_layer<ttl::nn::layers::conv<ttl::nn::ops::nhwc>>();
    // FIXME: support conv<nchw, rscd>
    // test_conv_layer<nn::layers::conv<ttl::nn::ops::nchw>>();
    test_conv_layer<
        ttl::nn::layers::conv<ttl::nn::ops::nchw, ttl::nn::ops::dcrs>>();

    test_pool_layer<ttl::nn::layers::pool<ttl::nn::ops::pool_max>>();
    test_pool_layer<
        ttl::nn::layers::pool<ttl::nn::ops::pool_max, ttl::nn::ops::nhwc>>();
    test_pool_layer<
        ttl::nn::layers::pool<ttl::nn::ops::pool_max, ttl::nn::ops::nchw>>();
}
