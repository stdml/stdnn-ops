#pragma once
#include <iomanip>
#include <iostream>

#include <ttl/debug>
#include <ttl/range>
#include <ttl/tensor>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r> class show_t
{
    const tensor_view<R, r> t_;

  public:
    show_t(const tensor_view<R, r> &t) : t_(t) {}

    operator tensor_view<R, r>() const { return t_; }
};

template <typename R, rank_t r>
std::ostream &operator<<(std::ostream &os, const show_t<R, r> &s)
{
    tensor_view<R, r> t = s;
    os << to_string(t.shape()) << std::endl;
    return os;
}

template <typename R>
std::ostream &operator<<(std::ostream &os, const show_t<R, 1> &s)
{
    tensor_view<R, 1> t = s;
    if (t.shape().size() < 10) {
        os << "[";
        for (auto i : ttl::range<0>(t)) {
            if (i) { os << ", "; }
            os << std::setw(6) << t.at(i);
        }
        os << "]" << std::endl;
    }

    return os;
}

template <typename R>
std::ostream &operator<<(std::ostream &os, const show_t<R, 2> &s)
{
    tensor_view<R, 2> t = s;
    os << to_string(t.shape()) << std::endl;

    if (t.shape().size() < 100) {
        for (auto i : ttl::range<0>(t)) {
            for (auto j : ttl::range<1>(t)) {
                if (j) { os << ", "; }
                os << std::setw(6) << t.at(i, j);
            }
            os << std::endl;
        }
    }

    return os;
}
}  // namespace internal
}  // namespace ttl
