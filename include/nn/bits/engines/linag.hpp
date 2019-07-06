#pragma once
#include <ttl/tensor>

#include <nn/bits/engines/config.hpp>

namespace nn::engines
{
template <typename E> struct linag {
    template <typename R> using m_ref_t = ttl::matrix_ref<R>;
    template <typename R> using m_view_t = ttl::matrix_view<R>;
    template <typename R> using v_ref_t = ttl::vector_ref<R>;
    template <typename R> using v_view_t = ttl::vector_view<R>;

    // a \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.m
    template <typename R>
    static void mm(const m_view_t<R> &a, const m_view_t<R> &b,
                   const m_ref_t<R> &c)
    {
        using engine = typename backend<E>::template type<R>;
        engine::mm(a, b, c);
    }

    // a \times b^T -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.n
    template <typename R>
    static void mmt(const m_view_t<R> &a, const m_view_t<R> &b,
                    const m_ref_t<R> &c)
    {
        using engine = typename backend<E>::template type<R>;
        engine::mmt(a, b, c);
    }

    // a^T \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.m == b.m
    template <typename R>
    static void mtm(const m_view_t<R> &a, const m_view_t<R> &b,
                    const m_ref_t<R> &c)
    {
        using engine = typename backend<E>::template type<R>;
        engine::mtm(a, b, c);
    }

    // a \times b -> c where a[m, n], b[n] -> c[n]; a.n = b.n
    template <typename R>
    static void mv(const m_view_t<R> &a, const v_view_t<R> &b,
                   const v_ref_t<R> &c)
    {
        using engine = typename backend<E>::template type<R>;
        engine::mv(a, b, c);
    }

    // [1, n] X [n, m] -> [1, m]
    // a \times b -> c where a[n], b[m, n] -> c[n]; a.n = b.m
    template <typename R>
    static void vm(const v_view_t<R> &a, const m_view_t<R> &b,
                   const v_ref_t<R> &c)
    {
        using engine = typename backend<E>::template type<R>;
        engine::vm(a, b, c);
    }
};
}  // namespace nn::engines
