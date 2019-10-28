#include <nn/bits/ops/transpose.hpp>
#include <nn/ops>
#include <nn/testing>

void test_transpose(int n, int h, int w, int c)
{
    ttl::tensor<int, 4> x(n, h, w, c);
    ttl::tensor<int, 4> x1(x.shape());

    ttl::tensor<int, 4> y(n, c, h, w);
    ttl::tensor<int, 4> y1(y.shape());

    const int d = x.shape().size();
    for (int i = 0; i < d; ++i) { x.data()[i] = i; }

    {
        nn::ops::to_channels_first()(ref(y), view(x));
        nn::ops::to_channels_last()(ref(x1), view(y));
        for (int i = 0; i < d; ++i) { ASSERT_EQ(x.data()[i], x1.data()[i]); }

        nn::ops::to_channels_first()(ref(y1), view(x1));
        for (int i = 0; i < d; ++i) { ASSERT_EQ(y.data()[i], y1.data()[i]); }
    }

    {
        nn::ops::to_channels_first()(ref(y), view(x),
                                     [](int x) { return x + 1; });
        nn::ops::to_channels_last()(ref(x1), view(y),
                                    [](int x) { return x - 1; });
        for (int i = 0; i < d; ++i) { ASSERT_EQ(x.data()[i], x1.data()[i]); }

        nn::ops::to_channels_first()(ref(y1), view(x1));
        for (int i = 0; i < d; ++i) {
            ASSERT_EQ(y.data()[i] - 1, y1.data()[i]);
        }
    }
}

TEST(transpose_test, test_1)
{
    std::array<int, 3> dims({3, 4, 5});
    do {
        const auto [h, w, c] = dims;
        test_transpose(2, h, w, c);
    } while (std::next_permutation(dims.begin(), dims.end()));
}
