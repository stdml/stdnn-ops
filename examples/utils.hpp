#include <iomanip>
#include <iostream>
#include <string>

#include <stdtensor>

#include <nn/ops>

#include <experimental/range>

template <ttl::rank_t r>
std::string show_shape(const ttl::internal::basic_shape<r> &shape,
                       char bracket_l = '(', char bracket_r = ')')
{
    std::string ss;
    for (auto i : std::experimental::range(r)) {
        if (!ss.empty()) { ss += ", "; }
        ss += std::to_string(shape.dims[i]);
    }
    return bracket_l + ss + bracket_r;
}

template <typename T, typename... Ts>
void show_signature(const T &y, const Ts &... x)
{
    std::array<std::string, sizeof...(Ts)> args({show_shape(x.shape())...});
    std::string ss;
    for (const auto &p : args) {
        if (!ss.empty()) { ss += ", "; }
        ss += p;
    }
    const auto sign = show_shape(y.shape()) + " <- " + ss;
    printf("%s\n", sign.c_str());
}

template <typename T> void pprint(const T &t, const char *name)
{
    printf("%s :: %s\n", name, show_shape(t.shape()).c_str());
}

#define PPRINT(e) pprint(e, #e);

namespace nn::layers::debug
{
class show_name
{
    const std::string name_;

  public:
    show_name(const std::string name) : name_(name) {}

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &y) const
    {
        std::cerr << std::left << std::setw(32) << name_
                  << " :: " << show_shape(y.shape()) << std::endl;
    }
};
}  // namespace nn::layers::debug

inline void make_unuse(void *) {}

#define UNUSED(e)                                                              \
    {                                                                          \
        make_unuse(&e);                                                        \
    }
