#pragma once
#include <experimental/bits/contract.hpp>

#include <nn/bits/ops/traits.hpp>

namespace nn::ops
{

template <typename image_order>
shape<4> batch_norm_infer(const shape<4> &x, const shape<1> &r_mean,
                          const shape<1> &r_var)
{
    const auto c = ops::channel_size<image_order>(x);
    contract_assert(c == std::get<0>(r_mean.dims));
    contract_assert(c == std::get<0>(r_var.dims));
    return x;
}

template <typename image_order> class batch_norm;

template <> class batch_norm<nhwc>
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &r_mean,
                        const shape<1> &r_var) const
    {
        return batch_norm_infer<nhwc>(x, r_mean, r_var);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &rolling_mean,
                    const ttl::tensor_view<R, 1> &rolling_var) const
    {
        constexpr R eps = .000001f;

        const auto [n, h, w, c] = x.shape().dims;

        for (const auto l : range(n)) {
            for (const auto i : range(h)) {
                for (const auto j : range(w)) {
                    for (const auto k : range(c)) {
                        y.at(l, i, j, k) =
                            (x.at(l, i, j, k) - rolling_mean.at(k)) /
                            (std::sqrt(rolling_var.at(k)) + eps);
                    }
                }
            }
        }
    }
};

template <> class batch_norm<nchw>
{
  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &r_mean,
                        const shape<1> &r_var) const
    {
        return batch_norm_infer<nchw>(x, r_mean, r_var);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &rolling_mean,
                    const ttl::tensor_view<R, 1> &rolling_var) const
    {
        constexpr R eps = .000001f;

        const auto [n, c, h, w] = x.shape().dims;

        for (const auto l : range(n)) {
            for (const auto k : range(c)) {
                for (const auto i : range(h)) {
                    for (const auto j : range(w)) {
                        y.at(l, k, i, j) =
                            (x.at(l, k, i, j) - rolling_mean.at(k)) /
                            (std::sqrt(rolling_var.at(k)) + eps);
                    }
                }
            }
        }
    }
};

template <typename image_order> class batch_norm_with_bias
{
    using bn_op = batch_norm<image_order>;

  public:
    shape<4> operator()(const shape<4> &x, const shape<1> &r_mean,
                        const shape<1> &r_var, const shape<1> &beta,
                        const shape<1> &gamma) const
    {
        const auto c = ops::channel_size<image_order>(x);
        contract_assert(c == std::get<0>(r_mean.dims));
        contract_assert(c == std::get<0>(r_var.dims));
        contract_assert(c == std::get<0>(beta.dims));
        contract_assert(c == std::get<0>(gamma.dims));
        return x;
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 1> &rolling_mean,
                    const ttl::tensor_view<R, 1> &rolling_var,
                    const ttl::tensor_view<R, 1> &beta,
                    const ttl::tensor_view<R, 1> &gamma) const
    {
        using add_bias = apply_bias<image_order, std::plus<R>>;
        using mul_bias = apply_bias<image_order, std::multiplies<R>>;
        bn_op()(y, x, rolling_mean, rolling_var);
        mul_bias()(y, view(y), gamma);
        add_bias()(y, view(y), beta);
    }
};

}  // namespace nn::ops
