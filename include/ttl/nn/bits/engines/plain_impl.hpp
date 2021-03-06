#pragma once
#include <cstring>

#include <algorithm>
#include <experimental/contract>
#include <numeric>

#include <ttl/range>
#include <ttl/tensor>

namespace ttl::nn::engines
{
template <typename T>
class plain_impl
{
    using m_ref_t = ttl::matrix_ref<T>;
    using m_view_t = ttl::matrix_view<T>;
    using v_ref_t = ttl::vector_ref<T>;
    using v_view_t = ttl::vector_view<T>;

  public:
    // a \times b -> c where a[m, l], b[l, n] -> c[m, n]
    static void mm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        const auto [m, l] = a.dims();
        const auto [_l, n] = b.dims();
        contract_assert(_l == l);
        contract_assert(m == std::get<0>(c.dims()));
        contract_assert(n == std::get<1>(c.dims()));

        std::memset(c.data(), 0, c.data_size());
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
        const auto [m, l] = a.dims();
        const auto [n, _l] = b.dims();
        contract_assert(_l == l);
        contract_assert(m == std::get<0>(c.dims()));
        contract_assert(n == std::get<1>(c.dims()));

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
        const auto [l, m] = a.dims();
        const auto [_l, n] = b.dims();
        contract_assert(_l == l);
        contract_assert(m == std::get<0>(c.dims()));
        contract_assert(n == std::get<1>(c.dims()));

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
        const auto [m, n] = a.dims();
        contract_assert(n == std::get<0>(b.dims()));
        contract_assert(m == std::get<0>(c.dims()));

        for (auto i : range(m)) {
            T tmp = 0;
            for (auto j : range(n)) { tmp += a.at(i, j) * b.at(j); }
            c.at(i) = tmp;
        }
    }

    // a \times b -> c where a[m], b[m, n] -> c[n]
    static void vm(const v_view_t &a, const m_view_t &b, const v_ref_t &c)
    {
        const auto [m, n] = b.dims();
        contract_assert(m == std::get<0>(a.dims()));
        contract_assert(n == std::get<0>(c.dims()));

        for (auto i : range(n)) {
            T tmp = 0;
            for (auto j : range(m)) { tmp += a.at(j) * b.at(j, i); }
            c.at(i) = tmp;
        }
    }
};
}  // namespace ttl::nn::engines
