#include <nn/bits/ops/init.hpp>

#include "testing.hpp"

TEST(init_test, test1)
{
    ttl::tensor<float, 2> x(2, 5);
    nn::ops::uniform_distribution()(ref(x));
    for (auto i : range(x.shape().size())) {
        ASSERT_FLOAT_EQ(x.data()[i], 0.1);
    }
}
