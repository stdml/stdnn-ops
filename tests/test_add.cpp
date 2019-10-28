#include <ttl/algorithm>
#include <ttl/tensor>

#include <nn/ops>
#include <nn/testing>

TEST(add_test, test1)
{
    const uint32_t n = 10;
    const uint32_t c = 3;
    const uint32_t h = 100;
    const uint32_t w = 100;

    using add = nn::ops::add;
    const auto x = ttl::tensor<int, 4>(n, h, w, c);
    const auto y = ttl::tensor<int, 4>(n, h, w, c);
    ttl::fill(ref(x), 1);
    ttl::fill(ref(y), 2);

    const auto op = add();
    const auto z = ttl::tensor<int, 4>(op(x.shape(), y.shape()));
    ASSERT_EQ(z.shape(), ttl::internal::basic_shape<4>(n, h, w, c));

    op(ref(z), view(x), view(y));

    ASSERT_EQ(nn::ops::summaries::sum()(view(z)), (int)(n * h * w * c * 3));
}
