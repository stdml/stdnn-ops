#include <ttl/tensor>

#include <nn/ops>
#include <nn/testing>

TEST(im2col_test, test1)
{
    using im2col = nn::ops::im2col<nn::ops::hw, nn::ops::hwrs>;
    const auto op = im2col(im2col::ksize(3, 5));

    auto x = ttl::tensor<int, 2>(32, 64);
    auto y = ttl::tensor<int, 4>(op(x.shape()));
    ASSERT_EQ(y.shape(), nn::shape<4>(30, 60, 3, 5));
    op(ref(y), view(x));
}

TEST(im2col_test, test2)
{
    using im2col = nn::ops::im2col<nn::ops::hw, nn::ops::rshw>;
    const auto op = im2col(im2col::ksize(3, 5));

    auto x = ttl::tensor<int, 2>(32, 64);
    auto y = ttl::tensor<int, 4>(op(x.shape()));
    ASSERT_EQ(y.shape(), nn::shape<4>(3, 5, 30, 60));
    op(ref(y), view(x));
}
