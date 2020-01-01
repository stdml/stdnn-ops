#include <ttl/nn/bits/ops/hash.hpp>
#include <ttl/nn/testing>

TEST(hash_test, test_crc32)
{
    const std::string s("Hello world");
    const ttl::tensor_view<char, 1> x(s.c_str(), s.size());
    const ttl::nn::ops::crc<> crc32q(0xD5828281);
    ttl::tensor<uint32_t, 0> y;
    crc32q(ref(y), x);
    ASSERT_EQ(y.data()[0], static_cast<uint32_t>(0x2964d064));
}
