#pragma once
#include <algorithm>

#include <ttl/range>
#include <ttl/tensor>

#include <nn/ops>

#include <gtest/gtest.h>

using ttl::range;

template <typename R, ttl::rank_t r>
void gen_test_tensor(const ttl::tensor<R, r> &x, int &s)
{
    std::generate(x.data(), x.data() + x.shape().size(), [&s] {
        s = (s + 7) % 10;
        return static_cast<R>(s);
    });
}

template <typename R> struct assert_eq {
    void operator()(const R &x, const R &y) { ASSERT_EQ(x, y); }
};

template <> struct assert_eq<float> {
    void operator()(float x, float y) { ASSERT_FLOAT_EQ(x, y); }
};

template <> struct assert_eq<double> {
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

inline void unused(void *) {}

#define UNUSED(e)                                                              \
    {                                                                          \
        unused(&e);                                                            \
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

template <typename T> void pprint(const T &t, const char *name)
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
    for (auto i : range(n)) {
        if (i > 0) { printf(", "); }
        printf("%s", show_scalar(x.data()[i]).c_str());
    }
    printf("}\n");
}

#define PT(e) show_tensor(e, #e);
