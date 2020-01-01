#include <ttl/algorithm>
#include <ttl/nn/bits/ops/col2im.hpp>
#include <ttl/nn/testing>

TEST(col2im_test, test1)
{
    using image_order = ttl::nn::ops::hwc;
    using col_order = ttl::nn::ops::hwrsc;

    using C = ttl::nn::ops::im2col_trait<ttl::nn::ops::hw>;

    using F = ttl::nn::ops::im2col<image_order, col_order>;
    using G = ttl::nn::ops::col2im<image_order, col_order>;

    using R = int;

    const int h = 3;
    const int w = 3;
    const int c = 1;
    const int r = 2;
    const int s = 2;

    const F f(C::ksize(r, s));
    const G g(C::ksize(r, s));

    ttl::shape<3> x_shape(h, w, c);
    ttl::shape<5> y_shape(f(x_shape));

    // TODO: support reverse shape infer
    // ttl::shape<3> x_shape_(g(y_shape));
    // ASSERT_EQ(x_shape_, x_shape);

    ttl::tensor<R, 3> x(x_shape);  // i32(3,3,1)
    ttl::tensor<R, 5> y(y_shape);

    {
        std::iota(x.data(), x.data_end(), 0);
        f(ref(y), view(x));
        std::vector<int> result({
            0, 1, 3, 4,  //
            1, 2, 4, 5,  //
            3, 4, 6, 7,  //
            4, 5, 7, 8   //
        });

        const uint32_t l = 16;
        ASSERT_EQ(y.shape().size(), l);
        ASSERT_EQ(static_cast<uint32_t>(result.size()), l);
        for (auto i : range(l)) { ASSERT_EQ(result[i], y.data()[i]); }
    }

    {
        ttl::fill(ref(y), static_cast<R>(1));
        g(ref(x), view(y));
        std::vector<int> result({
            1, 2, 1,  //
            2, 4, 2,  //
            1, 2, 1,  //
        });
        const uint32_t l = 9;
        ASSERT_EQ(x.shape().size(), l);
        ASSERT_EQ(static_cast<uint32_t>(result.size()), l);
        for (auto i : range(l)) { ASSERT_EQ(result[i], x.data()[i]); }
    }
}
