#pragma once
#include <cstring>

#include <algorithm>
#include <experimental/contract>
#include <numeric>

#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::engines
{
template <typename T> struct plain_impl {
    using m_ref_t = ttl::matrix_ref<T>;
    using m_view_t = ttl::matrix_view<T>;
    using v_ref_t = ttl::vector_ref<T>;
    using v_view_t = ttl::vector_view<T>;

    // a \times b -> c where a[m, l], b[l, n] -> c[m, n]
    static void mm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        const auto [m, l] = a.shape().dims();
        const auto [_l, n] = b.shape().dims();
        contract_assert(_l == l);
        contract_assert(m == c.shape().dims()[0]);
        contract_assert(n == c.shape().dims()[1]);

        std::memset(c.data(), 0, sizeof(T) * c.shape().size());
        for (auto i : range(m)) {
            for (auto k : range(l)) {
                const T aa = a.at(i, k);
                for (auto j : range(n)) { c.at(i, j) += aa * b.at(k, j); }
            }
        }
    }

    // a \times b^T -> c where a[m, l], b[n, l] -> c[m, n]
    static void mmt(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        const auto [m, l] = a.shape().dims();
        const auto [n, _l] = b.shape().dims();
        contract_assert(_l == l);
        contract_assert(m == c.shape().dims()[0]);
        contract_assert(n == c.shape().dims()[1]);

        for (auto i : range(m)) {
            for (auto j : range(n)) {
                T tmp = 0;
                for (auto k : range(l)) { tmp += a.at(i, k) * b.at(j, k); }
                c.at(i, j) = tmp;
            }
        }
    }

    // a^T \times b -> c where a[l, m], b[l, n] -> c[m, n]
    static void mtm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        const auto [l, m] = a.shape().dims();
        const auto [_l, n] = b.shape().dims();
        contract_assert(_l == l);
        contract_assert(m == c.shape().dims()[0]);
        contract_assert(n == c.shape().dims()[1]);

        for (auto i : range(m)) {
            for (auto j : range(n)) {
                T tmp = 0;
                for (auto k : range(l)) { tmp += a.at(k, i) * b.at(k, j); }
                c.at(i, j) = tmp;
            }
        }
    }

    // a \times b -> c where a[m, n], b[n] -> c[m]
    static void mv(const m_view_t &a, const v_view_t &b, const v_ref_t &c)
    {
        const auto [m, n] = a.shape().dims();
        contract_assert(n == b.shape().dims()[0]);
        contract_assert(m == c.shape().dims()[0]);

        for (auto i : range(m)) {
            T tmp = 0;
            for (auto j : range(n)) { tmp += a.at(i, j) * b.at(j); }
            c.at(i) = tmp;
        }
    }

    // a \times b -> c where a[m], b[m, n] -> c[n]
    static void vm(const v_view_t &a, const m_view_t &b, const v_ref_t &c)
    {
        const auto [m, n] = b.shape().dims();
        contract_assert(m == a.shape().dims()[0]);
        contract_assert(n == c.shape().dims()[0]);

        for (auto i : range(n)) {
            T tmp = 0;
            for (auto j : range(m)) { tmp += a.at(j) * b.at(j, i); }
            c.at(i) = tmp;
        }
    }
};
}  // namespace ttl::nn::engines
