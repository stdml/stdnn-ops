#pragma once
#include <nn/bits/traits/filter.hpp>
#include <nn/bits/traits/padding.hpp>
#include <nn/common.hpp>

namespace nn
{
namespace traits
{
template <typename dim_t> class linear_sample_trait
{
  public:
    using filter_t = linear_filter_trait<dim_t>;
    using padding_t = linear_padding_trait<dim_t>;

  private:
    const filter_t filter_;
    const padding_t padding_;

  public:  // static
    static constexpr dim_t default_rate = 1;
    static constexpr dim_t default_stride = 1;
    static constexpr dim_t default_pad_lr = 0;

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

    using padding_policy = std::function<padding_t(dim_t, dim_t, dim_t, dim_t)>;

    // maybe TODO: make it a value instead of a function
    static padding_policy padding_valid()
    {
        return padding_policy(valid_padding);
    }

    // maybe TODO: make it a value instead of a function
    static padding_policy padding_same()
    {
        return padding_policy(same_padding);
    }

    static padding_policy padding_fixed(const padding_t &p)
    {
        return [p = p](dim_t k, dim_t s, dim_t r, dim_t n) { return p; };
    }

  public:  // constructors
    linear_sample_trait(const dim_t ksize, const dim_t stride = 1,
                        const dim_t rate = 1, const dim_t pad_lr = 0)
        : filter_(ksize, stride, rate), padding_(pad_lr)
    {
    }

    linear_sample_trait(const dim_t ksize, const dim_t stride, const dim_t rate,
                        const dim_t pad_l, const dim_t pad_r)
        : filter_(ksize, stride, rate), padding_(pad_l, pad_r)
    {
    }

    linear_sample_trait(const dim_t ksize, const dim_t stride, const dim_t rate,
                        const padding_t &padding)
        : filter_(ksize, stride, rate), padding_(padding)
    {
    }

  public:                                                // getters
    dim_t get_ksize() const { return filter_.ksize(); }  // FIXME: deprecate

    dim_t get_stride() const { return filter_.stride(); }  // FIXME: deprecate

    padding_t get_padding() const { return padding_; }

  public:
    dim_t operator()(dim_t n) const { return filter_(n, padding_); }

    dim_t operator()(dim_t j, dim_t k) const { return filter_(j, k); }

    bool inside(dim_t i, dim_t n) const { return padding_.inside(i, n); }

    dim_t unpad(dim_t i) const { return padding_.unpad(i); }
};
}  // namespace traits
}  // namespace nn
