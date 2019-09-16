#pragma once
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nn
{
namespace traits
{
template <typename N> class linear_filter_trait
{
    const N ksize_;
    const N stride_;
    const N rate_;

  public:
    static constexpr N default_ksize = 1;
    static constexpr N default_stride = 1;
    static constexpr N default_rate = 1;

    linear_filter_trait(const N ksize = default_ksize,
                        const N stride = default_stride,
                        const N rate = default_rate)
        : ksize_(ksize), stride_(stride), rate_(rate)
    {
        // static_assert(std::is_unsigned<N>::value, "");
        if (ksize < 1) {
            throw std::invalid_argument("invalid ksize: " +
                                        std::to_string(ksize));
        }
        if (stride < 1) {
            throw std::invalid_argument("invalid stride: " +
                                        std::to_string(stride));
        }
        if (rate < 1) {
            throw std::invalid_argument("invalid rate: " +
                                        std::to_string(rate));
        }
    }

    const N ksize() const { return ksize_; }

    const N stride() const { return stride_; }

    const N rate() const { return rate_; }

    const N patch_size() const { return (ksize_ - 1) * rate_ + 1; }

    template <typename dim_t, typename Padding>
    dim_t operator()(dim_t n, const Padding &padding) const
    {
        const dim_t n_padded = padding(n);
        const dim_t ps = patch_size();
        if (n_padded < ps) {
            throw std::invalid_argument("linear sample input is too small: " +
                                        std::to_string(n));
        }
        if ((n_padded - ps) % stride_) {
            throw std::invalid_argument("invalid padding");
        }
        return (n_padded - ps) / stride_ + 1;
    }

    template <typename dim_t> dim_t operator()(dim_t j, dim_t k) const
    {
        // j^\prime = j * stride_;
        // patch_offset = k * rate_;
        return j * stride_ + k * rate_;
    }
};
}  // namespace traits
}  // namespace nn
