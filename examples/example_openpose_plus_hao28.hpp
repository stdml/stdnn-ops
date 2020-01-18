#pragma once
#include <ttl/nn/layers>
#include <ttl/nn/models>
#include <ttl/nn/ops>

// https://github.com/tensorlayer/openpose-plus/blob/master/openpose_plus/models/models_hao28_experimental.py
template <typename R> struct openpose_plus_hao28_impl {
    using image_order = ttl::nn::ops::nhwc;
    using filter_order = ttl::nn::ops::rscd;
    using relu = ttl::nn::ops::pointwise<ttl::nn::ops::relu>;
    using pool = ttl::nn::layers::pool<ttl::nn::ops::pool_max, image_order>;
    using concat = ttl::nn::ops::concat_channel4d<image_order>;

    mutable nn::models::namescope ns;
    const std::string data_dir_;

    auto f(const std::string &name) const
    {
        const auto full = data_dir_ + "/" + ns(name) + ".idx";
        return ttl::nn::ops::readfile(full);
    };

    auto conv(int d, int k, int p, const std::string &name = "conv") const
    {
        return ns.with(name, [&] {
            using conv_layer =
                ttl::nn::layers::conv<image_order, filter_order, false>;
            const auto l = conv_layer(d, conv_layer::ksize(k, k),
                                      conv_layer::padding_same());
            return with_init(l, f("kernel"));
        });
    }

    auto conv_(int d, int k, int p, const std::string &name = "conv") const
    {
        return ns.with(name, [&] {
            using conv_layer =
                ttl::nn::layers::conv<image_order, filter_order, true>;
            const auto l = conv_layer(d, conv_layer::ksize(k, k),
                                      conv_layer::padding_same());
            return with_init(l, f("kernel"), f("bias"));
        });
    }

    auto bn(const std::string &name = "bn") const
    {
        return ns.with(name, [&] {
            using relu = ttl::nn::ops::pointwise<ttl::nn::ops::relu>;
            using bn_layer = ttl::nn::layers::batch_norm<image_order, relu>;
            return with_init(bn_layer(), f("moving_mean"), f("moving_variance"),
                             f("beta"), f("gamma"));
        });
    }

  public:
    openpose_plus_hao28_impl(const std::string prefix) : data_dir_(prefix) {}

    const int n_joins = 19;
    const int n_connections = 19;

    auto operator()(const ttl::tensor_ref<R, 4> &x)
    {
        return ns.with("model", [&] {
            const auto fm = cnn(x);
            auto p1 = stage1(*fm);
            auto p5 = [&] {
                return ns.with("stage5", [&] {
                    return stage2(*fm, *p1.first, *p1.second);
                });
            }();
            auto p6 = [&] {
                return ns.with("stage6", [&] {
                    return stage2(*fm, *p5.first, *p5.second);
                });
            }();
            return p6;
        });
    }

  private:
    auto cnn(const ttl::tensor_ref<R, 4> &x)
    {
        auto conv_layers =                                //
            ttl::nn::models::make_sequential()            //
            << conv(32, 3, 1, "conv1_1") << bn("bn1_1")   //
            << conv(64, 3, 1, "conv1_2") << bn("bn1_2")   //
            << pool()                                     //
            << conv(128, 3, 1, "conv2_1") << bn("bn2_1")  //
            << conv(128, 3, 1, "conv2_2") << bn("bn2_2")  //
            << pool()                                     //
            << conv(200, 3, 1, "conv3_1") << bn("bn3_1")  //
            << conv(200, 3, 1, "conv3_2") << bn("bn3_2")  //
            << conv(200, 3, 1, "conv3_3") << bn("bn3_3")  //
            << pool()                                     //
            << conv(384, 3, 1, "conv4_1") << bn("bn4_1")  //
            << conv(384, 3, 1, "conv4_2") << bn("bn4_2")  //
            << conv(256, 3, 1, "conv4_3") << bn("bn4_3")  //
            << conv(128, 3, 1, "conv4_4") << bn("bn4_4")  //
            ;
        return conv_layers(x);
    }

    auto stage1(const ttl::tensor_ref<R, 4> &x)
    {
        return ns.with("stage1", [&] {
            auto common = [&] {
                return                                     //
                    ttl::nn::models::make_sequential()     //
                    << conv(128, 3, 1, "c1") << bn("bn1")  //
                    << conv(128, 3, 1, "c2") << bn("bn2")  //
                    << conv(128, 3, 1, "c3") << bn("bn3")  //
                    << conv(128, 1, 0, "c4") << bn("bn4")  //
                    ;
            };
            auto left = [&] {
                return ns.with("branch1", [&] {
                    return common() << conv_(n_joins, 1, 0, "confs");
                });
            }();
            auto right = [&] {
                return ns.with("branch2", [&] {
                    return common() << conv_(2 * n_connections, 1, 0, "pafs");
                });
            }();
            return std::make_pair(left(x), right(x));
        });
    }

    auto stage2(const ttl::tensor_ref<R, 4> &x,  //
                const ttl::tensor_ref<R, 4> &b1,
                const ttl::tensor_ref<R, 4> &b2)
    {
        using T = ttl::tensor<R, 4>;
        auto net = std::unique_ptr<T>(
            ttl::nn::ops::new_result<T>(concat(), x, b1, b2));
        auto common = [&] {
            return                                     //
                ttl::nn::models::make_sequential()     //
                << conv(128, 3, 1, "c1") << bn("bn1")  //
                << conv(128, 3, 1, "c2") << bn("bn2")  //
                << conv(128, 3, 1, "c3") << bn("bn3")  //
                << conv(128, 3, 1, "c4") << bn("bn4")  //
                << conv(128, 3, 1, "c5") << bn("bn5")  //
                << conv(128, 1, 0, "c6") << bn("bn6")  //
                ;
        };
        auto left = [&] {
            return ns.with("branch1", [&] {
                return common() << conv_(n_joins, 1, 0, "conf");
            });
        }();
        auto right = [&] {
            return ns.with("branch2", [&] {
                return common() << conv_(2 * n_connections, 1, 0, "pafs");
            });
        }();
        return std::make_pair(left(ref(*net)), right(ref(*net)));
    }
};

struct openpose_plus_hao28 {
    const std::string data_dir_;

  public:
    const size_t h;
    const size_t w;

    openpose_plus_hao28(const std::string &data_dir, int height = 256,
                        int width = 384)
        : data_dir_(data_dir), h(height), w(width)
    {
    }

    template <typename R> auto operator()(const ttl::tensor_ref<R, 4> &x)
    {
        return openpose_plus_hao28_impl<R>(data_dir_)(x);
    }
};
