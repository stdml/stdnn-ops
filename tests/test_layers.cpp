#include <nn/layers>

#include "testing.hpp"

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
        conv_layer l2(32, conv_layer::ksize(3, 3), conv_layer::padding(1, 1));
        l2(ref(x));
    }

    {
        auto x = ttl::tensor<float, 4>(2, 224, 224, 224);
        using conv_trait = nn::ops::conv_trait<nn::ops::hw>;
        conv_layer l1(
            1, conv_layer::ksize(7, 7),
            conv_trait(conv_trait::padding(conv_trait::padding_1d(3, 2),
                                           conv_trait::padding_1d(3, 2)),
                       conv_trait::stride(2, 2)));
        auto l = l1(ref(x));
        const auto shp = (*l).shape();
        ASSERT_EQ(shp.size(), 2 * 112 * 112);
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
    test_dense_layer<nn::layers::dense<>>();
    test_dense_layer<nn::layers::dense<nn::ops::pointwise<nn::ops::relu>>>();

    test_conv_layer<nn::layers::conv<>>();
    test_conv_layer<nn::layers::conv<nn::ops::nhwc>>();
    // FIXME: support conv<nchw, rscd>
    // test_conv_layer<nn::layers::conv<nn::ops::nchw>>();
    test_conv_layer<nn::layers::conv<nn::ops::nchw, nn::ops::dcrs>>();

    test_pool_layer<nn::layers::pool<nn::ops::pool_max>>();
    test_pool_layer<nn::layers::pool<nn::ops::pool_max, nn::ops::nhwc>>();
    test_pool_layer<nn::layers::pool<nn::ops::pool_max, nn::ops::nchw>>();
}
