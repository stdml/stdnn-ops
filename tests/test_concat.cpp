#include <ttl/nn/ops>
#include <ttl/nn/testing>

TEST(concat_test, test_1)
{
    using ttl::nn::ops::internal::concat_shape;
    using s = ttl::shape<2>;
    {
        auto u = s(3, 4);
        auto v = s(5, 4);
        {
            auto w = concat_shape<0>(u, v);
            ASSERT_EQ(w, s(8, 4));
        }
        {
            auto w = concat_shape<0>(u, v, u);
            ASSERT_EQ(w, s(11, 4));
        }
        {
            auto w = concat_shape<0>(u, v, u, v);
            ASSERT_EQ(w, s(16, 4));
        }
    }
    {
        auto u = s(3, 4);
        auto v = s(3, 5);
        {
            auto w = concat_shape<1>(u, v);
            ASSERT_EQ(w, s(3, 9));
        }
        {
            auto w = concat_shape<1>(u, v, u);
            ASSERT_EQ(w, s(3, 13));
        }
        {
            auto w = concat_shape<1>(u, v, u, v);
            ASSERT_EQ(w, s(3, 18));
        }
    }
}

TEST(concat_test, test_2)
{
    const int n = 10;
    const int h = 8;
    const int w = 8;
    using T = ttl::tensor<int, 4>;
    {
        ttl::nn::ops::concat_channel4d<ttl::nn::traits::nhwc> concat;
        T x(n, h, w, 3);
        T y(n, h, w, 4);
        T z(n, h, w, 5);
        T t(n, h, w, 12);
        check_infer(concat, t, x, y, z);

        std::iota(x.data(), x.data_end(), 0);
        std::iota(y.data(), y.data_end(), x.size());
        std::iota(z.data(), z.data_end(), x.size() + y.size());
        concat(ttl::ref(t), ttl::view(z), ttl::view(x), ttl::view(y));
        ASSERT_TRUE(fast_is_permutation(ttl::flatten(ttl::view(t))));
    }
    {
        ttl::nn::ops::concat_channel4d<ttl::nn::traits::nchw> concat;
        T x(n, 3, h, w);
        T y(n, 4, h, w);
        T t(n, 7, h, w);
        check_infer(concat, t, x, y);

        std::iota(x.data(), x.data_end(), 0);
        std::iota(y.data(), y.data_end(), x.size());
        concat(ttl::ref(t), ttl::view(x), ttl::view(y));
        ASSERT_TRUE(fast_is_permutation(ttl::flatten(ttl::view(t))));
    }
}
