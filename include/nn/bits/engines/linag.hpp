#pragma once
#include <nn/bits/engines/config.hpp>

namespace nn::engines
{
template <typename T, typename engine = default_engine<T>> struct linag {
    using m_ref_t = ttl::matrix_ref<T>;
    using m_view_t = ttl::matrix_view<T>;
    using v_ref_t = ttl::vector_ref<T>;
    using v_view_t = ttl::vector_view<T>;

    // a \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.m
    static void mm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        engine::mm(a, b, c);
    }

    // a \times b^T -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.n
    static void mmt(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        engine::mmt(a, b, c);
    }

    // a^T \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.m == b.m
    static void mtm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        engine::mtm(a, b, c);
    }

    // a \times b -> c where a[m, n], b[n] -> c[n]; a.n = b.n
    static void mv(const m_view_t &a, const v_view_t &b, const v_ref_t &c)
    {
        engine::mv(a, b, c);
    }

    // [1, n] X [n, m] -> [1, m]
    // a \times b -> c where a[n], b[m, n] -> c[n]; a.n = b.m
    static void vm(const v_view_t &a, const m_view_t &b, const v_ref_t &c)
    {
        engine::vm(a, b, c);
    }

    // a + b -> c
    static void vv(const v_view_t &a, const v_view_t &b, const v_ref_t &c)
    {
        engine::vv(a, b, c);
    }
};
}  // namespace nn::engines
