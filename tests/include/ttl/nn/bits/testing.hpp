#pragma once
#include <algorithm>

#include <ttl/nn/ops>  // FIXME: don't include
#include <ttl/range>
#include <ttl/tensor>

#include <gtest/gtest.h>

template <typename R, ttl::rank_t r>
void gen_test_tensor(const ttl::tensor<R, r> &x, int &s)
{
    std::generate(x.data(), x.data() + x.shape().size(), [&s] {
        s = (s + 7) % 10;
        return static_cast<R>(s);
    });
}

template <typename R>
struct assert_eq {
    void operator()(const R &x, const R &y) { ASSERT_EQ(x, y); }
};

template <>
struct assert_eq<float> {
    void operator()(float x, float y) { ASSERT_FLOAT_EQ(x, y); }
};

template <>
struct assert_eq<double> {
    void operator()(double x, double y) { ASSERT_FLOAT_EQ(x, y); }
};

template <typename T, typename... I>
int test_all_permutations(const T &t, I... i)
{
    std::array<int, sizeof...(I)> a({static_cast<int>(i)...});
    std::sort(a.begin(), a.end());
    int p = 0;
    do {
        std::apply(t, a);
        ++p;
    } while (std::next_permutation(a.begin(), a.end()));
    return p;
}

template <typename R, ttl::rank_t r>
void assert_tensor_eq(const ttl::tensor_view<R, r> &x,
                      const ttl::tensor_view<R, r> &y)
{
    ASSERT_EQ(x.shape(), y.shape());
    for (auto i : ttl::range(x.shape().size())) {
        assert_eq<R>()(x.data()[i], y.data()[i]);
    }
}

template <typename R, ttl::rank_t r>
ttl::tensor_view<uint8_t, 1> view_bytes(const ttl::tensor_view<R, r> &x)
{
    return ttl::tensor_view<uint8_t, 1>(
        reinterpret_cast<const uint8_t *>(x.data()), x.data_size());
}

template <typename R, ttl::rank_t r>
bool bytes_eq(const ttl::tensor_view<R, r> &x, const ttl::tensor_view<R, r> &y)
{
    if (x.shape() != y.shape()) { return false; }
    const auto a = view_bytes(x);
    const auto b = view_bytes(y);
    return std::equal(a.data(), a.data_end(), b.data());
}

template <typename R, ttl::rank_t r>
void assert_bytes_eq(const ttl::tensor_view<R, r> &x,
                     const ttl::tensor_view<R, r> &y)
{
    ASSERT_TRUE(bytes_eq(x, y));
}

template <typename T>
void pprint(const T &t, const char *name)
{
    printf("%s :: %s\n", name, std::to_string(t.shape()).c_str());
}

#define PPRINT(e) pprint(e, #e);

inline std::string show_scalar(int x) { return std::to_string(x); }

template <typename R, ttl::rank_t r>
void show_tensor(const ttl::tensor<R, r> &x, const std::string name = "t")
{
    pprint(x, name.c_str());
    const auto n = x.shape().size();
    printf("  data: {");
    for (auto i : ttl::range(n)) {
        if (i > 0) { printf(", "); }
        printf("%s", show_scalar(x.data()[i]).c_str());
    }
    printf("}\n");
}

#define PT(e) show_tensor(e, #e);

template <typename F, typename Y, typename... Xs>
void check_infer(const F &f, const Y &y, const Xs &... xs)
{
    const auto shape = f(xs.shape()...);
    ASSERT_EQ(shape, y.shape());
}

template <typename N>
bool fast_is_permutation(const ttl::tensor_view<N, 1> &x)
{
    const N n = x.size();
    std::vector<bool> b(n);
    std::fill(b.begin(), b.end(), false);
    for (auto i : ttl::range(n)) {
        const N j = x.data()[i];
        if (j < 0 || n <= j || b[j]) { return false; }
        b[j] = true;
    }
    return true;
}
