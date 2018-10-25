#pragma once
#include <experimental/contract>

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
    static constexpr dim_t default_rate = 1;
    static constexpr dim_t default_stride = 1;
    static constexpr dim_t default_pad_lr = 0;

    const dim_t pad_l_;  // TODO: make it template parameter
    const dim_t pad_r_;  // TODO: make it template parameter

  public:
    //   FIXME:
    const dim_t rate_;
    const dim_t stride_;
    const dim_t ksize_;

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
        : linear_sample_trait(ksize, stride, rate, pad_lr, pad_lr)
    {
    }

    linear_sample_trait(dim_t ksize, dim_t stride, dim_t rate, dim_t pad_l,
                        dim_t pad_r)
        : pad_l_(pad_l),
          pad_r_(pad_r),
          rate_(rate),
          stride_(stride),
          ksize_(ksize)
    {
        contract_assert(rate_ >= 1);
        contract_assert(stride_ >= 1);
        contract_assert(ksize_ >= 1);
        contract_assert(pad_l >= 0);
        contract_assert(pad_r >= 0);
    }

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

}  // namespace nn::ops
