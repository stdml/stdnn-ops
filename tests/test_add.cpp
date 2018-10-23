#include <gtest/gtest.h>

#include <nn/ops>
#include <stdtensor>

#include "testing.hpp"

TEST(add_test, test1)
{
    const uint32_t n = 10;
    const uint32_t c = 3;
    const uint32_t h = 100;
    const uint32_t w = 100;

    using add = nn::ops::add;
    const auto x = ttl::tensor<int, 4>(n, h, w, c);
    const auto y = ttl::tensor<int, 4>(n, h, w, c);
    fill(x, 1);
    fill(y, 2);

    const auto op = add();
    const auto z = ttl::tensor<int, 4>(op(x.shape(), y.shape()));
    ASSERT_EQ(z.shape(), ttl::internal::basic_shape<4>(n, h, w, c));

    op(ref(z), view(x), view(y));

    ASSERT_EQ(sum(z), (int)(n * h * w * c * 3));
}
