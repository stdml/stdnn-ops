#include <ttl/nn/bits/ops/hash.hpp>
#include <ttl/nn/testing>

TEST(hash_test, test_crc32)
{
    const std::string s("Hello world");
    const ttl::tensor_view<char, 1> x(s.c_str(), s.size());
    {
        const ttl::nn::ops::crc<> crc32q(0xD5828281);
        ttl::tensor<uint32_t, 0> y;
        crc32q(ref(y), x);
        ASSERT_EQ(y.data()[0], static_cast<uint32_t>(0x2964d064));
    }
    {
        const ttl::nn::ops::crc<> crc32ieee;
        ttl::tensor<uint32_t, 0> y;
        crc32ieee(ref(y), x);
        ASSERT_EQ(y.data()[0], static_cast<uint32_t>(0x8bd69e52));
    }
}

TEST(hash_test, test_crc64)
{
    const std::string s("Hello world");
    const ttl::tensor_view<char, 1> x(s.c_str(), s.size());
    {
        const ttl::nn::ops::crc<uint64_t> crc64ecma;
        ttl::tensor<uint64_t, 0> y;
        crc64ecma(ref(y), x);
        ASSERT_EQ(y.data()[0], static_cast<uint64_t>(0xf4a5f2b9d47756bf));
    }
}
