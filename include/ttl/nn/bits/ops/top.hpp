#pragma once
#include <algorithm>
#include <vector>

#include <ttl/tensor>

#include <ttl/nn/common.hpp>

namespace nn::ops
{
class top
{
    using dim_t = nn::shape<1>::dimension_type;

    const dim_t k_;

  public:
    top(int k) : k_(k) {}

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &y,
                    const ttl::tensor_ref<dim_t, 1> &z,
                    const ttl::tensor_view<R, 1> &x) const
    {
        const dim_t n = x.shape().size();

        using vt = std::vector<std::pair<R, dim_t>>;
        vt v(n);
        for (auto i : range(n)) {
            v[i].first = x.at(i);
            v[i].second = i;
        }
        std::sort(v.begin(), v.end(), std::greater<typename vt::value_type>());

        for (auto i : range(std::min(n, k_))) {
            y.at(i) = v[i].first;
            z.at(i) = v[i].second;
        }
    }
};
}  // namespace nn::ops
