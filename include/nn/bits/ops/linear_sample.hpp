#pragma once
#include <nn/common.hpp>

/*!
\begin{definition}
A Linear sample is a map that maps a sequence of length $n$ to a sequence of
length $m$.
\end{definition}


The sample takes the following steps
\begin{itemize}
\item The input sequence $x[n]$ is padded to $x^\prime[pad_l + n + pad_r]$ by
prepending $pad_l$ zero elements and appending $pad_r$ zero elements.

\item The kernel (filter) $y[k]$ is extended to $y^\prime[rate * (k - 1) + 1]$
by inserting $rate - 1$ zero elements to adjacent elemnts.

\item The padded input $x^\prime$ and the extended kernel $y^\prime$ takes a
valid convolution with stride $s$, resulting the output sequence $Z[m]$,
where $m = (n^\prime - k^prime) / s + 1$, and $s \mid n^\prime - k^prime$ should
be granted.
\end{itemize}


e.g.

012345678901234567890123456789
0123456789ABCDEF0123456789ABCD // n = 30, pad_l = pad_r = 0
*   *   *                      // ksize = 3, rate = 4, patch_size = 9
       *   *   *               // stride = 7
              *   *   *
                     *   *   *
                               // m = (30 - 9) / 7 + 1 = 4

*/

namespace nn::ops
{

template <typename dim_t> class linear_sample_trait
{
    const dim_t pad_l_;  // TODO: make it template parameter
    const dim_t pad_r_;  // TODO: make it template parameter

    const dim_t rate_;
    const dim_t stride_;
    const dim_t ksize_;

    struct padding_trait;

  public:
    static constexpr dim_t default_rate = 1;
    static constexpr dim_t default_stride = 1;
    static constexpr dim_t default_pad_lr = 0;

    using padding_t = std::experimental::new_type<shape<2>, padding_trait>;

    static padding_t padding(dim_t p) { return padding_t(p, p); }

    static padding_t padding(dim_t left, dim_t right)
    {
        return padding_t(left, right);
    };

  public:
    linear_sample_trait(dim_t ksize)
        : linear_sample_trait(ksize, default_stride)
    {
    }

    linear_sample_trait(dim_t ksize, dim_t stride)
        : linear_sample_trait(ksize, stride, default_rate)
    {
    }

    linear_sample_trait(dim_t ksize, dim_t stride, dim_t rate)
        : linear_sample_trait(ksize, stride, rate, default_pad_lr)
    {
    }

    linear_sample_trait(dim_t ksize, dim_t stride, dim_t rate, dim_t pad_lr)
        : linear_sample_trait(ksize, stride, rate, padding(pad_lr))
    {
    }

    linear_sample_trait(dim_t ksize, dim_t stride, dim_t rate, dim_t pad_l,
                        dim_t pad_r)
        : linear_sample_trait(ksize, stride, rate, padding(pad_l, pad_r))
    {
        // TODO: deprecate it
    }

    linear_sample_trait(dim_t ksize, dim_t stride, dim_t rate,
                        const padding_t &pad)
        : pad_l_(std::get<0>(pad.dims)),
          pad_r_(std::get<1>(pad.dims)),
          rate_(rate),
          stride_(stride),
          ksize_(ksize)
    {
        contract_assert(rate_ >= 1);
        contract_assert(stride_ >= 1);
        contract_assert(ksize_ >= 1);
        contract_assert(pad_l_ >= 0);
        contract_assert(pad_r_ >= 0);
    }

    dim_t get_ksize() const { return ksize_; }

    dim_t get_stride() const { return stride_; }

    /*! Compute the output size from input size. */
    dim_t operator()(dim_t n) const
    {
        const dim_t n_padded = n + pad_l_ + pad_r_;
        const dim_t patch_size = rate_ * (ksize_ - 1) + 1;
        contract_assert(n_padded >= patch_size);
        contract_assert((n_padded - patch_size) % stride_ == 0);
        return (n_padded - patch_size) / stride_ + 1;
    }

    /*! Compute the index $i^\prime$ of padded input from the output index and
     * the kernel index, the original index $i$ should be $i^prime - pad_l$ if
     * $pad_l <= i^\prime <  pad_l + n$.
     *
     *
     * identities:
     *      (0, 0) -> 0
     *      (m - 1, ksize_ - 1) -> n - 1
     * */
    // requires: 0 <= j < m
    // requires: 0 <= k < ksize_
    dim_t operator()(dim_t j, dim_t k) const
    {
        // j^\prime = j * stride_;
        // patch_offset = k * rate_;
        return j * stride_ + k * rate_;
    }

    bool inside(dim_t i, dim_t n) const
    {
        return pad_l_ <= i && i < pad_l_ + n;
    }

    dim_t unpad(dim_t i) const { return i - pad_l_; }
};

namespace internal
{
template <size_t idx, typename T> static T constant(const T &x) { return x; }

template <typename C, typename T, std::size_t... I>
static auto replicate_construct(const T &x, std::index_sequence<I...>)
{
    return C(constant<I>(x)...);
}

template <typename T, std::size_t... I>
static std::array<T, sizeof...(I)> replicate(const T &x,
                                             std::index_sequence<I...>)
{
    return std::array<T, sizeof...(I)>({constant<I>(x)...});
}

}  // namespace internal

template <ttl::rank_t r, typename dim_t> class multi_linear_sample_trait
{
  protected:
    using sample_t = linear_sample_trait<dim_t>;
    using padding_1d_t = typename sample_t::padding_t;

    struct ksize_trait;
    struct stride_trait;
    struct rate_trait;

    using ksize_t = std::experimental::new_type<shape<r>, ksize_trait>;
    using stride_t = std::experimental::new_type<shape<r>, stride_trait>;
    using rate_t = std::experimental::new_type<shape<r>, rate_trait>;
    using padding_t = std::array<padding_1d_t, r>;

    const std::array<sample_t, r> samples_;

    static stride_t default_stride()
    {
        return internal::replicate_construct<stride_t>(
            sample_t::default_stride, std::make_index_sequence<r>());
    }

    static rate_t default_rate()
    {
        return internal::replicate_construct<rate_t>(
            sample_t::default_rate, std::make_index_sequence<r>());
    }

    static padding_t default_padding()
    {
        return internal::replicate(padding_1d(sample_t::default_pad_lr),
                                   std::make_index_sequence<r>());
    }

  public:
    static padding_1d_t padding_1d(dim_t p) { return padding_1d_t(p, p); }

    static padding_1d_t padding_1d(dim_t left, dim_t right)
    {
        return padding_1d_t(left, right);
    }

    template <typename... D> static padding_t padding_simple(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return padding(padding_1d(static_cast<dim_t>(d))...);
    };

    template <typename... D> static padding_t padding(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return padding(padding_1d(static_cast<dim_t>(d))...);
    };

    template <typename... Padding1D>
    static padding_t padding(const padding_1d_t &p1 /* de-ambiguity */,
                             const Padding1D &... p)
    {
        static_assert(sizeof...(Padding1D) == r - 1,
                      "invalid number of arguments");
        return {p1, static_cast<padding_1d_t>(p)...};
    };

    template <typename... D> static ksize_t ksize(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return ksize_t(d...);
    };

    template <typename... D> static stride_t stride(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return stride_t(d...);
    };

    template <typename... D> static rate_t rate(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return rate_t(d...);
    };

    multi_linear_sample_trait(const ksize_t &ksize)
        : multi_linear_sample_trait(ksize, default_padding(), default_stride())
    {
    }

    multi_linear_sample_trait(const ksize_t &ksize, const padding_t &padding)
        : multi_linear_sample_trait(ksize, padding, default_stride())
    {
    }

    multi_linear_sample_trait(const ksize_t &ksize, const stride_t &stride)
        : multi_linear_sample_trait(ksize, default_padding(), stride)
    {
    }

    multi_linear_sample_trait(const ksize_t &ksize, const padding_t &padding,
                              const stride_t &stride)
        : multi_linear_sample_trait(ksize, padding, stride, default_rate())
    {
    }

    multi_linear_sample_trait(const ksize_t &ksize, const padding_t &padding,
                              const stride_t &stride, const rate_t &rate)
        : samples_(construct(ksize, padding, stride, rate,
                             std::make_index_sequence<r>()))
    {
    }

    template <typename... Sample>
    multi_linear_sample_trait(const Sample &... sample) : samples_(sample...)
    {
    }

    shape<r> operator()(const shape<r> &x) const
    {
        return invoke(x, std::make_index_sequence<r>());
    }

  private:
    template <std::size_t... I>
    static std::array<sample_t, r>
    construct(const ksize_t &ksize, const padding_t &padding,
              const stride_t &stride, const rate_t &rate,
              std::index_sequence<I...>)
    {
        static_assert(sizeof...(I) == r, "");
        return {sample_t(std::get<I>(ksize.dims), std::get<I>(stride.dims),
                         std::get<I>(rate.dims), std::get<I>(padding))...};
    }

    template <std::size_t... I>
    shape<r> invoke(const shape<r> &x, std::index_sequence<I...>) const
    {
        static_assert(sizeof...(I) == r, "");
        return shape<r>(std::get<I>(samples_)(std::get<I>(x.dims))...);
    }
};

}  // namespace nn::ops
