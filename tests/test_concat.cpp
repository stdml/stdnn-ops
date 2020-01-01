#include <ttl/nn/ops>
#include <ttl/nn/testing>

using nn::ops::internal::concat_shape;

TEST(concat_test, test_1)
{
    {
        using s = nn::shape<2>;
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
}
