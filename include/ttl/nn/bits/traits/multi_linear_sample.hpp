#pragma once
#include <ttl/nn/bits/traits/linear_sample.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/shape>

namespace ttl
{
namespace nn
{
namespace traits
{
namespace internal
{
template <size_t idx, typename T>
static T constant(const T &x)
{
    return x;
}

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

template <rank_t r, typename dim_t>
class multi_linear_sample_trait
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

  public:  // static
    static padding_1d_t padding_1d(dim_t p) { return padding_1d_t(p, p); }

    static padding_1d_t padding_1d(dim_t left, dim_t right)
    {
        return padding_1d_t(left, right);
    }

    template <typename... D>
    static padding_t padding_simple(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return padding(padding_1d(static_cast<dim_t>(d))...);
    };

    template <typename... D>
    static padding_t padding(D... d)
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

    template <typename... D>
    static ksize_t ksize(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return ksize_t(d...);
    };

    template <typename... D>
    static stride_t stride(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return stride_t(d...);
    };

    template <typename... D>
    static rate_t rate(D... d)
    {
        static_assert(sizeof...(D) == r, "invalid number of arguments");
        return rate_t(d...);
    };

  public:  // constructors
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
        static_assert(sizeof...(Sample) == r);
    }

  public:
    ksize_t get_ksize() const
    {
        return get_ksize(std::make_index_sequence<r>());
    }

    shape<r> operator()(const shape<r> &x) const
    {
        return invoke(x, std::make_index_sequence<r>());
    }

    // const auto &samples() const { return samples_; }

  private:
    template <std::size_t... I>
    static std::array<sample_t, r>
    construct(const ksize_t &ksize, const padding_t &padding,
              const stride_t &stride, const rate_t &rate,
              std::index_sequence<I...>)
    {
        static_assert(sizeof...(I) == r, "");
        return {sample_t(std::get<I>(ksize.dims()), std::get<I>(stride.dims()),
                         std::get<I>(rate.dims()), std::get<I>(padding))...};
    }

    template <std::size_t... I>
    shape<r> invoke(const shape<r> &x, std::index_sequence<I...>) const
    {
        static_assert(sizeof...(I) == r, "");
        return shape<r>(std::get<I>(samples_)(std::get<I>(x.dims()))...);
    }

    template <std::size_t... I>
    ksize_t get_ksize(std::index_sequence<I...>) const
    {
        static_assert(sizeof...(I) == r, "");
        return ksize_t(std::get<I>(samples_).get_ksize()...);
    }
};
}  // namespace traits
}  // namespace nn
}  // namespace ttl
