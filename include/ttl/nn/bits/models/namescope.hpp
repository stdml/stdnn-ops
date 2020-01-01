#pragma once
#include <string>
#include <vector>

namespace nn::models
{
template <typename ctx_t> class name_scope_t_
{
  public:
    name_scope_t_(const std::string &name, ctx_t &ctx) : name(name), ctx(ctx)
    {
        ctx.in(name);
    }

    ~name_scope_t_() { ctx.out(name); }

  private:
    const std::string name;
    ctx_t &ctx;
};

class name_prefix_ctx_t
{
  public:
    explicit name_prefix_ctx_t(const std::string &sep = "/") : sep_(sep) {}

    ~name_prefix_ctx_t() {}

    void in(const std::string &name) { names_.push_back(name); }

    void out(const std::string &name) { names_.pop_back(); }

    std::string operator*() const
    {
        std::string ss;
        for (const auto &s : names_) {
            if (!ss.empty()) { ss += sep_; }
            ss += s;
        }
        return ss;
    }

    std::string operator()(const std::string &name) const
    {
        std::string ss = this->operator*();
        if (!ss.empty()) { ss += sep_; }
        return ss + name;
    }

    template <typename F> auto with(const std::string &prefix, const F &f)
    {
        name_scope_t_<name_prefix_ctx_t> _(prefix, *this);
        return f();
    }

  private:
    std::string sep_;
    std::vector<std::string> names_;
};

using namescope = name_prefix_ctx_t;

}  // namespace nn::models
