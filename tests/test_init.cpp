#include <ttl/nn/bits/ops/hash.hpp>
#include <ttl/nn/bits/ops/init.hpp>
#include <ttl/nn/testing>

TEST(init_test, test_01)
{
    ttl::tensor<int, 2> x(2, 5);
    ttl::tensor<int, 2> y(2, 5);
    ttl::nn::ops::zeros()(ttl::ref(x));
    ttl::nn::ops::ones()(ttl::ref(y));
    for (auto i : ttl::range(x.shape().size())) {
        ASSERT_FLOAT_EQ(x.data()[i], 0);
    }
    for (auto i : ttl::range(y.shape().size())) {
        ASSERT_FLOAT_EQ(y.data()[i], 1);
    }
}

TEST(init_test, test_uniform)
{
    ttl::tensor<float, 2> x(2, 5);
    ttl::nn::ops::uniform_constant()(ttl::ref(x));
    for (auto i : ttl::range(x.shape().size())) {
        ASSERT_FLOAT_EQ(x.data()[i], 0.1);
    }
}

TEST(init_test, test_truncated_normal)
{
    const ttl::nn::ops::truncated_normal init(0.1);
    ttl::tensor<float, 2> x(2, 5);
    ttl::tensor<float, 2> y(x.shape());
    ttl::tensor<uint32_t, 0> seed;
    seed.data()[0] = 0;

    init(ref(x));
    init(ref(y));
    init(ref(x), view(seed));
    init(ref(y), view(seed));

    assert_bytes_eq(view(x), view(y));

    ttl::tensor<uint32_t, 0> h;
    ttl::tensor<uint32_t, 0> k;

    const ttl::nn::ops::crc<> crc32;
    crc32(ref(h), view(x));
    crc32(ref(k), view(y));
    ASSERT_EQ(h.data()[0], k.data()[0]);
}
