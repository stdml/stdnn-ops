#include <ttl/tensor>

#include <nn/ops>

#include "testing.hpp"

using nn::shape;
using nn::ops::as_mat_shape;

TEST(reshape_test, test1)
{
    const shape<6> s(1, 2, 3, 4, 5, 6);

    const auto s0 = as_mat_shape<0, 6>(s);
    ASSERT_EQ(s0, shape<2>(1, 720));

    const auto s1 = as_mat_shape<1, 5>(s);
    ASSERT_EQ(s1, shape<2>(1, 720));

    const auto s2 = as_mat_shape<2, 4>(s);
    ASSERT_EQ(s2, shape<2>(2, 360));

    const auto s3 = as_mat_shape<3, 3>(s);
    ASSERT_EQ(s3, shape<2>(6, 120));

    const auto s4 = as_mat_shape<4, 2>(s);
    ASSERT_EQ(s4, shape<2>(24, 30));

    const auto s5 = as_mat_shape<5, 1>(s);
    ASSERT_EQ(s5, shape<2>(120, 6));

    const auto s6 = as_mat_shape<6, 0>(s);
    ASSERT_EQ(s6, shape<2>(720, 1));
}
