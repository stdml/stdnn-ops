#include <gtest/gtest.h>

#include <nn/ops>
#include <stdtensor>

#include <nn/bits/ops/linear_sample.hpp>

#include "testing.hpp"

void test_linear_sample_ksr_nm(int ksize, int stride, int rate, int n, int m)
{
    nn::ops::linear_sample_trait sample(ksize, stride, rate);

    ASSERT_EQ(sample(n), m);
    ASSERT_EQ(sample(0, 0), 0);
    ASSERT_EQ(sample(m - 1, ksize - 1), n - 1);

    std::vector<int> x(n);
    std::fill(x.begin(), x.end(), 0);
    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < ksize; ++k) { x[sample(j, k)] += 1; }
    }
    ASSERT_EQ(std::accumulate(x.begin(), x.end(), 0), m * ksize);
    ASSERT_EQ(x[0], 1);
    ASSERT_EQ(x[n - 1], 1);
}

void test_linear_sample_ksr_m(int k, int s, int r, int m)
{
    // patch_size = (k - 1) * rate + 1
    // n = (m - 1) * stride + patch_size = (m - 1) * stride + (k - 1) * rate + 1
    int n = (m - 1) * s + (k - 1) * r + 1;
    test_linear_sample_ksr_nm(k, s, r, n, m);
}

void test_linear_sample_1()
{
    /*
    k, s, r = 3, 7, 4
    0123456789ABCDEF0123456789ABCD // n = 30, pad_l = pad_r = 0
    *   *   *                      // ksize = 3, rate = 4, patch_size = 9
           *   *   *               // stride = 7
                  *   *   *
                         *   *   *
                                   // m = (30 - 9) / 7 + 1 = 4
    */
    test_linear_sample_ksr_nm(3, 7, 4, 30, 4);

    for (int k = 1; k <= 3; ++k) {
        for (int s = 1; s <= 3; ++s) {
            for (int r = 1; r <= 3; ++r) {
                for (int m = 1; m <= 3; ++m) {
                    test_linear_sample_ksr_m(k, s, r, m);
                }
            }
        }
    }
}

TEST(linear_sample_test, test_1) { test_linear_sample_1(); }
