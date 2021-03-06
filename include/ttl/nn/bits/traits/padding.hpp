#pragma once
#include <array>
#include <stdexcept>

namespace ttl
{
namespace nn
{
namespace traits
{
template <typename N> class linear_padding_trait
{
    const N left_;
    const N right_;

  public:
    static constexpr N default_pad_lr = 0;

    linear_padding_trait(const N lr = default_pad_lr) : left_(lr), right_(lr) {}

    linear_padding_trait(const N l, const N r) : left_(l), right_(r) {}

    template <typename dim_t> dim_t operator()(dim_t n) const
    {
        return left_ + n + right_;
    }

    template <typename dim_t> bool inside(dim_t i, dim_t n) const
    {
        // FIXME: support negative padding
        return left_ <= static_cast<N>(i) &&
               static_cast<N>(i) < left_ + static_cast<N>(n);
    }

    template <typename dim_t> dim_t unpad(dim_t i) const { return i - left_; }

    const std::array<N, 2> dims() const  // FIXME: deprecate
    {
        return std::array<N, 2>({left_, right_});
    }
};
}  // namespace traits
}  // namespace nn
}  // namespace ttl
