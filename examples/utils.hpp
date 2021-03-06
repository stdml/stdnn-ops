#include <experimental/iterator>
#include <sstream>
#include <string>

#include <ttl/debug>
#include <ttl/nn/ops>
#include <ttl/tensor>

template <typename T, typename... Ts>
void show_signature(const T &y, const Ts &... x)
{
    const std::array<std::string, sizeof...(Ts)> args(
        {ttl::to_string(x.shape())...});
    std::stringstream ss;
    std::copy(args.begin(), args.end(),
              std::experimental::make_ostream_joiner(ss, ", "));
    const auto sign = ttl::to_string(y.shape()) + " <- " + ss.str();
    printf("%s\n", sign.c_str());
}

template <typename T>
void pprint(const T &t, const char *name)
{
    printf("%s :: %s\n", name, ttl::to_string(t.shape()).c_str());
}

#define PPRINT(e) pprint(e, #e);
