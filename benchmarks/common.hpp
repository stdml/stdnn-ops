#pragma once
#include <tuple>

#include <ttl/tensor>

namespace ttl
{
template <typename Tuple, size_t... I>
auto view_tuple(const Tuple &tup, std::index_sequence<I...>)
{
    return std::make_tuple(view(std::get<I>(tup))...);
}
}  // namespace ttl

template <typename F, typename R, typename S0, typename... Ss>
class bench
{
    static constexpr auto arity = sizeof...(Ss);

    using Y = ttl::tensor<R, S0::rank>;
    using Xs = std::tuple<ttl::tensor<R, Ss::rank>...>;

    F f;
    Y y;
    Xs xs;

  public:
    bench(const F &f, const Ss &... s)
        : f(f), y(f(s...)), xs(ttl::tensor<R, Ss::rank>(s)...)
    {
    }

    template <int i>
    const auto &input() const
    {
        return std::get<i>(xs);
    }

    template <int i, typename Init>
    void init(const Init &f) const
    {
        f(ttl::ref(input<i>()));
    }

    void operator()() const
    {
        const auto args = std::tuple_cat(
            std::make_tuple(ttl::ref(y)),
            ttl::view_tuple(xs, std::make_index_sequence<arity>()));
        std::apply(f, args);
    }
};
