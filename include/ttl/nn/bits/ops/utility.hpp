#pragma once
#include <ttl/nn/bits/kernels/cpu/utility.hpp>
#include <ttl/nn/bits/kernels/utility.hpp>
#include <ttl/nn/bits/ops/reshape.hpp>
#include <ttl/nn/common.hpp>

namespace ttl::nn::ops
{
class cast : public endofunction
{
  public:
    using endofunction::operator();

    template <typename R, typename S, rank_t r, typename D>
    void operator()(const tensor_ref<R, r, D> &y,
                    const tensor_view<S, r, D> &x) const
    {
        kernels::cast<D, R, S>()(flatten(y), flatten(x));
    }
};

class argmax : public reduce_function
{
  public:
    using reduce_function::operator();

    template <typename R, typename N, rank_t r, typename D>
    void operator()(const tensor_ref<N, r, D> &y,
                    const tensor_view<R, r + 1, D> &x) const
    {
        kernels::argmax<D, N, R>()(flatten(y), as_matrix<r, 1>(x));
    }
};

class onehot : public vectorize_function
{
  public:
    using vectorize_function::operator();

    onehot(const dim_t k) : vectorize_function(k) {}

    template <typename R, typename N, rank_t r, typename D>
    void operator()(const tensor_ref<R, r + 1, D> &y,
                    const tensor_view<N, r, D> &x) const
    {
        kernels::onehot<D, N, R>()(as_matrix<r, 1>(y), flatten(x));
    }
};

class similarity
{
  public:
    template <rank_t r>
    shape<0> operator()(const shape<r> &x, const shape<r> &y) const
    {
        contract_assert_eq(x, y);
        return shape<0>();
    }

    template <typename R, typename R1, rank_t r, typename D>
    void operator()(const tensor_ref<R, 0, D> &z,
                    const tensor_view<R1, r, D> &x,
                    const tensor_view<R1, r, D> &y) const
    {
        kernels::similarity<D, R, R1>()(z, flatten(x), flatten(y));
    }
};

class top
{
    using dim_t = shape<1>::dimension_type;

    const dim_t k_;

  public:
    top(int k) : k_(k) {}

    std::pair<shape<1>, shape<1>> operator()(const shape<1> &x) const
    {
        return std::make_pair(shape<1>(k_), shape<1>(k_));
    }

    template <typename R, typename N, typename D>
    void operator()(const tensor_ref<R, 1, D> &y, const tensor_ref<N, 1, D> &z,
                    const tensor_view<R, 1, D> &x) const
    {
        kernels::top<D, N, R>()(y, z, x);
    }
};
}  // namespace ttl::nn::ops
