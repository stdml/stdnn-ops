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

    void operator()() const { f_(); }

    T output() const { return y_; }
};

void call_all() {}

template <typename F1, typename... Fs>
void call_all(const F1 &f1, const Fs &... fs)
{
    f1();
    call_all(fs...);
}

class model_builder
{
    using raw_tensor = ttl::experimental::raw_tensor;
    using encoder = raw_tensor::encoder_type;

    std::vector<std::unique_ptr<raw_tensor>> data_;

    template <typename R, ttl::rank_t r>
    static operation<R, r> make_result(const ttl::tensor_ref<R, r> &y,
                                       const std::function<void()> &f)
    {
        return operation<R, r>(y, f);
    }

    //   public:
    template <typename R, ttl::rank_t r>
    ttl::tensor_ref<R, r> tensor(const nn::shape<r> &shape)
    {
        raw_tensor *t = new raw_tensor(encoder::value<float>(), shape);
        data_.push_back(std::unique_ptr<raw_tensor>(t));
        return ttl::tensor_ref<R, r>(reinterpret_cast<R *>(t->data()), shape);
    }

    template <typename R, typename... D>
    ttl::tensor_ref<R, sizeof...(D)> tensor(const D... d)
    {
        return tensor<R>(nn::shape<sizeof...(D)>(d...));
    }

    template <typename R, ttl::rank_t r>
    auto var(const ttl::tensor_ref<R, r> &t)
    {
        // TODO: check t is owned.
        return make_result(t, [] {});
    }

  public:
    template <typename R, typename... D> auto var(const D... d)
    {
        return var(tensor<R>(d...));
    }

    template <typename R, typename... D> auto covar(const D... d)
    {
        return var(tensor<R>(d...));
    }

    template <typename R, ttl::rank_t r, ttl::rank_t r1>
    auto reshape(const operation<R, r> &o, const nn::shape<r1> &shape)
    {
        ttl::tensor_ref<R, r1> t(o.output().data(), shape);
        return make_result(t, [=] { o(); });
    }

    template <typename R, typename Op, typename... Args>
    auto apply(const Op &op, const Args &... args)
    {
        const auto shape = op(args.output().shape()...);
        const auto t = tensor<R>(shape);
        const auto f = [=] {
            // TODO: dedup
            call_all(args...);
            op(t, view(args.output())...);
        };
        return make_result(t, f);
    }
};
