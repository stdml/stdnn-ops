// http://www.netlib.org/blas/
#pragma once
#include <ttl/tensor>

#include <cblas.h>

namespace ttl::nn::engines
{
template <typename R> struct _cblas;

template <> struct _cblas<float> {
    static constexpr auto axpy = cblas_saxpy;
    static constexpr auto gemm = cblas_sgemm;
    static constexpr auto gemv = cblas_sgemv;
};

template <> struct _cblas<double> {
    static constexpr auto axpy = cblas_daxpy;
    static constexpr auto gemm = cblas_dgemm;
    static constexpr auto gemv = cblas_dgemv;
};

template <typename R> struct cblas_impl {
    static constexpr R alpha = 1;
    static constexpr R beta = 0;
    static constexpr int inc = 1;

    using m_ref_t = ttl::matrix_ref<R>;
    using m_view_t = ttl::matrix_view<R>;
    using v_ref_t = ttl::vector_ref<R>;
    using v_view_t = ttl::vector_view<R>;

    using blas = _cblas<R>;

    template <typename T> static int len(const T &t)
    {
        return std::get<0>(t.shape().dims());
    }

    template <typename T> static int wid(const T &t)
    {
        return std::get<1>(t.shape().dims());
    }

    static void _gemm(const m_view_t &a, bool trans_a,  //
                      const m_view_t &b, bool trans_b, const m_ref_t &c)
    {
        blas::gemm(CblasRowMajor,
                   trans_a ? CblasTrans : CblasNoTrans,  //
                   trans_b ? CblasTrans : CblasNoTrans,  //
                   len(c), wid(c),                       //
                   trans_b ? wid(b) : len(b),            //
                   alpha, a.data(), wid(a), b.data(), wid(b), beta, c.data(),
                   wid(c));
    }

    // a \times b -> c
    static void mm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        _gemm(a, false, b, false, c);
    }

    // a \times b^T -> c
    static void mmt(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        _gemm(a, false, b, true, c);
    }

    // a^T \times b -> c
    static void mtm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        _gemm(a, true, b, false, c);
    }

    static void mv(const m_view_t &a, const v_view_t &b, const v_ref_t &c)
    {
        blas::gemv(CblasRowMajor, CblasNoTrans, len(a), wid(a), alpha, a.data(),
                   wid(a), b.data(), inc, beta, c.data(), inc);
    }

    static void vm(const v_view_t &a, const m_view_t &b, const v_ref_t &c)
    {
        blas::gemv(CblasRowMajor, CblasTrans, len(b), wid(b), alpha, b.data(),
                   wid(b), a.data(), inc, beta, c.data(), inc);
    }
};
}  // namespace ttl::nn::engines
