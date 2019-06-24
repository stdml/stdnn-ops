#pragma once
#include <memory>
#include <vector>

#include <ttl/tensor>

template <typename R, ttl::rank_t r> class operation
{
    using F = std::function<void()>;
    using T = ttl::tensor_ref<R, r>;

    const T y_;
    const F f_;

  public:
    operation(const T &y, const F &f) : y_(y), f_(f) {}

    T output() const { return y_; }
};

class model_builder
{
    // TODO: use ttl::experimental::raw_tensor instead of
    // std::unique_ptr<char[]>
    std::vector<std::unique_ptr<char[]>> data_;

    template <typename R, ttl::rank_t r>
    static operation<R, r> make_result(const ttl::tensor_ref<R, r> &y,
                                       const std::function<void()> &f)
    {
        return operation<R, r>(y, f);
    }

  public:
    template <typename R, ttl::rank_t r>
    ttl::tensor_ref<R, r> tensor(const nn::shape<r> &shape)
    {
        char *data = new char[sizeof(R) * shape.size()];
        data_.push_back(std::unique_ptr<char[]>(data));
        return ttl::tensor_ref<R, r>(reinterpret_cast<R *>(data), shape);
    }

    template <typename R, typename... D>
    ttl::tensor_ref<R, sizeof...(D)> tensor(const D... d)
    {
        return tensor<R>(nn::shape<sizeof...(D)>(d...));
    }

    template <typename R, ttl::rank_t r>
    auto var(const ttl::tensor_ref<R, r> &t)
    {
        return make_result(t, [] {});
    }

    template <typename R, typename... D> auto var(const D... d)
    {
        return var(tensor<R>(d...));
    }

    template <typename R, typename Op, typename... Args>
    auto apply(const Op &op, const Args &... args)
    {
        const auto shape = op(args.output().shape()...);
        const auto t = tensor<R>(shape);
        const auto f = [=] {
            // TODO: call args
            op(t, view(args.output())...);
        };
        return make_result(t, f);
    }
};
