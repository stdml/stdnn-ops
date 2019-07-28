#pragma once
#include <nn/bits/layers/call.hpp>
#include <nn/bits/layers/layer.hpp>
#include <nn/bits/ops/bias.hpp>
#include <nn/bits/ops/bn.hpp>
#include <nn/bits/ops/noop.hpp>

namespace nn::layers
{

template <typename image_order, typename Act = ops::noop, bool with_bias = true>
class batch_norm;

template <typename image_order, typename Act>
class batch_norm<image_order, Act, false>
{
    using bn_op = nn::ops::batch_norm<image_order>;

  public:
    template <typename R,                     //
              typename MeanInit = ops::noop,  //
              typename VarInit = ops::noop>
    auto operator()(const ttl::tensor_ref<R, 4> &x,
                    const MeanInit &mean_init = MeanInit(),
                    const VarInit &var_init = VarInit()) const
    {
        using T1 = ttl::tensor<R, 1>;
        using T4 = ttl::tensor<R, 4>;
        const auto shp = shape<1>(ops::channel_size<image_order>(x));

        auto rolling_mean = ops::new_parameter<T1>(shp, mean_init);
        auto rolling_var = ops::new_parameter<T1>(shp, var_init);

        auto y = ops::new_result<T4>(bn_op(), x, *rolling_mean, *rolling_var);

        Act()(ref(*y), view(*y));
        return make_layer(y, rolling_mean, rolling_var);
    }
};

template <typename image_order, typename Act>
class batch_norm<image_order, Act, true>
{
    using bn_op = nn::ops::batch_norm_with_bias<image_order>;

  public:
    template <typename R,                     //
              typename MeanInit = ops::noop,  //
              typename VarInit = ops::noop,
              typename BetaInit = ops::noop,  //
              typename GammaInit = ops::noop>
    auto operator()(const ttl::tensor_ref<R, 4> &x,  //
                    const MeanInit &mean_init = MeanInit(),
                    const VarInit &var_init = VarInit(),
                    const BetaInit &beta_init = BetaInit(),
                    const GammaInit &gamma_init = GammaInit()) const
    {
        using T1 = ttl::tensor<R, 1>;
        using T4 = ttl::tensor<R, 4>;
        const auto shp = shape<1>(ops::channel_size<image_order>(x.shape()));

        auto rolling_mean = ops::new_parameter<T1>(shp, mean_init);
        auto rolling_var = ops::new_parameter<T1>(shp, var_init);
        auto beta = ops::new_parameter<T1>(shp, beta_init);
        auto gamma = ops::new_parameter<T1>(shp, gamma_init);

        auto y = ops::new_result<T4>(bn_op(), x, *rolling_mean, *rolling_var,
                                     *beta, *gamma);
        Act()(ref(*y), view(*y));
        return make_layer(y, rolling_mean, rolling_var, beta, gamma);
    }
};

}  // namespace nn::layers
