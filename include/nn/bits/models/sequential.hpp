#pragma once
#include <nn/layers>
#include <nn/ops>

namespace nn::layers
{

namespace internal
{
template <typename L, typename... Init> class with_init
{
    const L layer_;
    const std::tuple<Init...> init_;

    template <typename T, std::size_t... I>
    auto call_with_init(const T &x, std::index_sequence<I...>) const
    {
        return layer_(x, std::get<I>(init_)...);
    }

  public:
    with_init(const L &layer, const Init &... init)
        : layer_(layer), init_(init...)
    {
    }

    template <typename T> auto operator()(const T &x) const
    {
        return call_with_init(x, std::make_index_sequence<sizeof...(Init)>());
    }
};
}  // namespace internal

template <typename L, typename... Init>
internal::with_init<L, Init...> with_init(const L &layer, const Init &... init)
{
    return internal::with_init<L, Init...>(layer, init...);
}

}  // namespace nn::layers

namespace nn::models
{
template <typename F, typename G> class composed
{
    const F f_;
    const G g_;

  public:
    composed(const F &f, const G &g) : f_(f), g_(g) {}

    template <typename T> auto operator()(const T &x) const
    {
        auto y = f_(x);
        auto z = g_(ref(*y));
        return z;
    }
};

template <typename F, typename G> composed<F, G> compose(const F &f, const G &g)
{
    return composed<F, G>(f, g);
}

template <typename Op> class sequential;

template <typename Op = nn::layers::identity>
sequential<Op> make_sequential(const Op &op = Op())
{
    return sequential<Op>(op);
}

template <typename Op> class sequential
{
    const Op op_;

  public:
    sequential(const Op &op) : op_(op) {}

    template <typename T> auto operator()(const T &x) const { return op_(x); }

    template <typename L> auto operator<<(const L &l) const
    {
        return make_sequential(compose(op_, l));
    }
};
}  // namespace nn::models
