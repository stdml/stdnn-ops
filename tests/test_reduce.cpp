#include <ttl/nn/ops>
#include <ttl/nn/testing>
#include <ttl/tensor>

/*
0  1  2  |  3
3  4  5  | 12
6  7  8  | 21
9  10 11 | 30
-- -- --
18 22 26
*/

TEST(reduce_test, test1)
{
    const int m = 4;
    const int n = 3;

    ttl::tensor<int, 2> x(m, n);
    std::iota(x.data(), x.data_end(), 0);

    ttl::tensor<int, 1> y(m);
    ttl::tensor<int, 1> z(n);

    nn::ops::internal::inner_contraction(ref(y), view(x));
    ASSERT_EQ(y.data()[0], 3);
    ASSERT_EQ(y.data()[1], 12);
    ASSERT_EQ(y.data()[2], 21);
    ASSERT_EQ(y.data()[3], 30);

    nn::ops::internal::outter_contraction(ref(z), view(x));
    ASSERT_EQ(z.data()[0], 18);
    ASSERT_EQ(z.data()[1], 22);
    ASSERT_EQ(z.data()[2], 26);
}
