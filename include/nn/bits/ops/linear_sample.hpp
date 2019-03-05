#pragma once
#include <nn/common.hpp>

/*!
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
struct fixed_padding;
struct valid_padding;
struct same_padding;

template <typename dim_t, typename padding_policy = fixed_padding>
class linear_sample_trait;

template <typename dim_t> class linear_sample_trait<dim_t, fixed_padding>
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

    static constexpr padding_t padding(dim_t p) { return padding_t(p, p); }

    static constexpr padding_t padding(dim_t left, dim_t right)
    {
        return padding_t(left, right);
    };

    static dim_t patch_size(dim_t k, dim_t r) { return r * (k - 1) + 1; }

    static padding_t even_padding(dim_t p)
    {
        return padding_t(p / 2, p - p / 2);
    }

    static padding_t valid_padding(dim_t k, dim_t s, dim_t r, dim_t n)
    {
        const dim_t ps = patch_size(k, r);
        // p is the minimal p such that (n + p - ps) == 0 (mod s)
        // p == ps - n (mod s)
        return even_padding((ps + s - (n % s)) % s);
    }

    static padding_t same_padding(dim_t k, dim_t s, dim_t r, dim_t n)
    {
        const dim_t ps = patch_size(k, r);
        const dim_t n0 = n % s;
        // p = ps - s - (n % s)
        // TODO: support negative padding
        contract_assert(ps >= s + n0);
        return even_padding(ps - s - n0);
    }

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

    padding_t get_padding() const { return padding(pad_l_, pad_r_); }

    /*! Compute the output size from input size. */
    dim_t operator()(dim_t n) const
    {
        const dim_t n_padded = n + pad_l_ + pad_r_;
        const dim_t ps = patch_size(ksize_, rate_);
        contract_assert(n_padded >= ps);
        contract_assert((n_padded - ps) % stride_ == 0);
        return (n_padded - ps) / stride_ + 1;
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

template <typename dim_t> class linear_sample_trait<dim_t, valid_padding>
{
    const dim_t rate_;
    const dim_t stride_;
    const dim_t ksize_;
};

template <typename dim_t> class linear_sample_trait<dim_t, same_padding>
{
    const dim_t rate_;
    const dim_t stride_;
    const dim_t ksize_;
};

template <typename dim_t>
constexpr typename linear_sample_trait<dim_t>::padding_t pad(dim_t p)
{
    return typename linear_sample_trait<dim_t>::padding_t(p, p);
}

template <typename dim_t>
constexpr typename linear_sample_trait<dim_t>::padding_t pad(dim_t l, dim_t r)
{
    return typename linear_sample_trait<dim_t>::padding_t(l, r);
}

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
    multi_linear_sample_trait(const Sample &... sample) : samples_({sample...})
    {
    }

    ksize_t get_ksize() const
    {
        return get_ksize(std::make_index_sequence<r>());
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

    template <std::size_t... I>
    ksize_t get_ksize(std::index_sequence<I...>) const
    {
        static_assert(sizeof...(I) == r, "");
        return ksize_t(std::get<I>(samples_).get_ksize()...);
    }
};

}  // namespace nn::ops
