#pragma once
#include <ttl/nn/bits/traits/multi_linear_sample.hpp>
#include <ttl/nn/common.hpp>
#include <ttl/nn/traits>

namespace ttl::nn::ops
{
template <typename image_order>
class pool_trait;

template <>
class pool_trait<hw>
{
  protected:
    struct ksize_trait;
    struct stride_trait;

    using dim_t = uint32_t;
    using sample1d_t_ = linear_sample_trait<dim_t>;

    using padding_1d_t = typename sample1d_t_::padding_t;
    using padding_t = std::array<padding_1d_t, 2>;
    using ksize_t = std::experimental::new_type<shape<2>, ksize_trait>;
    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;

    static constexpr auto default_ksize = ksize_t(2, 2);

    const sample1d_t_ h_sample_;
    const sample1d_t_ w_sample_;

    ksize_t get_ksize() const
    {
        return ksize_t(h_sample_.get_ksize(), w_sample_.get_ksize());
    }

    stride_t get_stride() const
    {
        return stride_t(h_sample_.get_stride(), w_sample_.get_stride());
    }

    padding_t get_padding() const
    {
        return padding(h_sample_.get_padding(), w_sample_.get_padding());
    }

  public:
    using sample1d_t = sample1d_t_;

    static padding_1d_t padding_1d(dim_t p) { return padding_1d_t(p, p); }

    static padding_1d_t padding_1d(dim_t left, dim_t right)
    {
        return padding_1d_t(left, right);
    }

    static padding_t padding(const padding_1d_t &r, const padding_1d_t &s)
    {
        return {r, s};
    };

    static padding_t padding(dim_t r, dim_t s)
    {
        return padding(sample1d_t::padding(r), sample1d_t::padding(s));
    }

    static ksize_t ksize(dim_t r, dim_t s) { return ksize_t(r, s); }

    static stride_t stride(dim_t r, dim_t s) { return stride_t(r, s); }

    pool_trait() : pool_trait(default_ksize) {}

    pool_trait(const ksize_t &ksize)
        : pool_trait(ksize, stride(ksize.dims()[0], ksize.dims()[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const stride_t &stride)
        : pool_trait(ksize, padding(0, 0), stride)
    {
    }

    pool_trait(const ksize_t &ksize, const padding_t &padding)
        : pool_trait(ksize, padding, stride(ksize.dims()[0], ksize.dims()[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const padding_t &padding,
               const stride_t &stride)
        : h_sample_(ksize.dims()[0], stride.dims()[0], 1, std::get<0>(padding)),
          w_sample_(ksize.dims()[1], stride.dims()[1], 1, std::get<1>(padding))
    {
    }

    pool_trait(const pool_trait &t)
        : h_sample_(t.h_sample_), w_sample_(t.w_sample_)
    {
    }

    shape<2> operator()(const shape<2> &x) const
    {
        return shape<2>(h_sample_(x.dims()[0]), w_sample_(x.dims()[1]));
    }
};
}  // namespace ttl::nn::ops
