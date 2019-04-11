#pragma once
#include <cstring>

#include <algorithm>

#include <nn/bits/ops/reshape.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops
{
using std::experimental::range;

template <typename R, ttl::rank_t r>
void fill(const ttl::tensor_ref<R, r> &t, R val)
{
    std::fill(t.data(), t.data() + t.shape().size(), val);
}

template <typename R>
nn::shape<1>::dimension_type argmax(const ttl::tensor_view<R, 1> &t)
{
    return std::max_element(t.data(), t.data() + t.shape().size()) - t.data();
}

class onehot
{
    using dim_t = shape<0>::dimension_type;
    const dim_t k_;

  public:
    onehot(const dim_t k) : k_(k) {}

    // template <ttl::rank_t r> shape<r + 1> operator()(const shape<r> &s) const
    // {
    //     std::array<dim_t, r + 1> dims;
    //     std::copy(s.dims.begin(), s.dims.end(), dims.begin());
    //     dims[r] = k_;
    //     return shape<r + 1>(dims);
    // }

    template <typename R, typename N, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r + 1> &y,
                    const ttl::tensor_view<N, r> &x) const
    {
        std::memset(y.data(), 0, sizeof(R) * y.shape().size());
        const auto y_flat = nn::ops::as_matrix<r, 1, ttl::tensor_ref<R, 2>>(y);
        const auto n = x.shape().size();
        for (auto i : range(n)) {
            y_flat.at(i, x.data()[i]) = static_cast<R>(1);
        }
    }
};
}  // namespace nn::experimental::ops
