#include <ttl/tensor>

#include <ttl/nn/bits/ops/gradients/reshape.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/testing>

template <typename R, ttl::rank_t... rs> struct test_flatten {
    static constexpr ttl::rank_t in_rank =
        ttl::internal::int_seq_sum<ttl::rank_t, rs...>::value;
    static constexpr ttl::rank_t out_rank = sizeof...(rs);

    void operator()(const ttl::shape<in_rank> &x_shape) const
    {
        nn::ops::copy_flatten<rs...> f;
        nn::ops::grad::copy_flatten<rs...> g;

        ttl::tensor<R, in_rank> x(x_shape);
        ttl::tensor<R, out_rank> y(f(x_shape));
        ttl::tensor<R, out_rank> gy(y.shape());
        ttl::tensor<R, in_rank> gx(g(gy.shape(), y.shape(), x.shape()));

        std::iota(x.data(), x.data_end(), 1);
        std::iota(gy.data(), gy.data_end(), 2);

        f(ref(y), view(x));
        g(ref(gx), view(gy), view(y), view(x));

        assert_tensor_eq(view(flatten(y)), view(flatten(x)));
        assert_tensor_eq(view(flatten(gx)), view(flatten(gy)));
    }
};

TEST(reshape_test, test1)
{
    test_flatten<int, 1, 3>()(ttl::make_shape(10, 28, 28, 3));
    test_flatten<int, 2, 2>()(ttl::make_shape(10, 28, 28, 3));
    test_flatten<int, 3, 1>()(ttl::make_shape(10, 28, 28, 3));
}
