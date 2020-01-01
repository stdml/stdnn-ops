#include <ttl/nn/bits/ops/io.hpp>
#include <ttl/nn/testing>

template <typename T> void test_io(const T &x)
{
    const auto y = ttl::tensor<typename T::value_type, T::rank>(x.shape());

    const std::string filename = "x.idx";
    (nn::ops::writefile(filename))(view(x));
    (nn::ops::readfile(filename))(ref(y));

    for (auto i : range(x.shape().size())) {
        ASSERT_EQ(x.data()[i], y.data()[i]);
    }
}

TEST(io_test, test1)
{
    {
        const auto x = ttl::tensor<float, 4>(2, 3, 4, 5);
        std::iota(x.data(), x.data() + x.shape().size(), 0);
        test_io(x);
    }
    {
        const auto x = ttl::tensor<double, 4>(2, 3, 4, 5);
        std::iota(x.data(), x.data() + x.shape().size(), 0);
        test_io(x);
    }
    {
        const auto x = ttl::tensor<std::uint8_t, 4>(2, 3, 4, 5);
        std::iota(x.data(), x.data() + x.shape().size(), 0);
        test_io(x);
    }
    {
        const auto x = ttl::tensor<std::int8_t, 4>(2, 3, 4, 5);
        std::iota(x.data(), x.data() + x.shape().size(), 0);
        test_io(x);
    }
    {
        const auto x = ttl::tensor<std::int16_t, 4>(2, 3, 4, 5);
        std::iota(x.data(), x.data() + x.shape().size(), 0);
        test_io(x);
    }
    {
        const auto x = ttl::tensor<std::int32_t, 4>(2, 3, 4, 5);
        std::iota(x.data(), x.data() + x.shape().size(), 0);
        test_io(x);
    }
}
