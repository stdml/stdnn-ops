#pragma once
#include <algorithm>
#include <experimental/range>

#include <ttl/tensor>

#include <nn/ops>

#include <gtest/gtest.h>

using std::experimental::range;

template <typename R, ttl::rank_t r> void fill(const ttl::tensor<R, r> &t, R x)
{
    std::fill(t.data(), t.data() + t.shape().size(), x);
}

template <typename R, ttl::rank_t r>
void gen_test_tensor(const ttl::tensor<R, r> &x, int &s)
{
    std::generate(x.data(), x.data() + x.shape().size(), [&s] {
        s = (s + 7) % 10;
        return static_cast<R>(s);
    });
}

template <typename R> struct assert_eq;

template <> struct assert_eq<float> {
    void operator()(float x, float y) { ASSERT_FLOAT_EQ(x, y); }
};

template <> struct assert_eq<double> {
    void operator()(double x, double y) { ASSERT_FLOAT_EQ(x, y); }
};

template <typename T>
int test_all_3_permutations(const T &t, int k, int m, int n)
{
    int p = 0;
    std::array<int, 3> a({k, m, n});
    std::sort(a.begin(), a.end());
    do {
        const auto [k, m, n] = a;
        t(k, m, n);
        ++p;
    } while (std::next_permutation(a.begin(), a.end()));
    return p;
}

inline void unused(void *) {}

#define UNUSED(e)                                                              \
    {                                                                          \
        unused(&e);                                                            \
    }

template <typename R, ttl::rank_t r>
void assert_tensor_eq(const ttl::tensor<R, r> &x, const ttl::tensor<R, r> &y)
{
    ASSERT_EQ(x.shape(), y.shape());
    const auto n = x.shape().size();
    for (auto i : range(n)) { ASSERT_EQ(x.data()[i], y.data()[i]); }
}

template <ttl::rank_t r>
std::string show_shape(const ttl::internal::basic_shape<r> &shape,
                       char bracket_l = '(', char bracket_r = ')')
{
    std::string ss;
    for (auto i : std::experimental::range(r)) {
        if (!ss.empty()) { ss += ", "; }
        ss += std::to_string(shape.dims[i]);
    }
    return bracket_l + ss + bracket_r;
}

template <typename T> void pprint(const T &t, const char *name)
{
    printf("%s :: %s\n", name, show_shape(t.shape()).c_str());
}

#define PPRINT(e) pprint(e, #e);

inline std::string show_scalar(int x) { return std::to_string(x); }

template <typename R, ttl::rank_t r>
void show_tensor(const ttl::tensor<R, r> &x, const std::string name = "t")
{
    pprint(x, name.c_str());
    const auto n = x.shape().size();
    printf("  data: {");
    for (auto i : range(n)) {
        if (i > 0) { printf(", "); }
        printf("%s", show_scalar(x.data()[i]).c_str());
    }
    printf("}\n");
}

#define PT(e) show_tensor(e, #e);
