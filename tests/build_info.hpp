#include <cstdio>
#include <sstream>
#include <string>

#include <nn/bits/engines/linag.hpp>

namespace _build_info
{
// http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system

constexpr bool is_linux =
#if defined(__linux__)
    true
#else
    false
#endif
    ;

constexpr bool is_mac =
#if defined(__APPLE__) && defined(__MACH__)
    true
#else
    false
#endif
    ;

// http://nadeausoftware.com/articles/2012/10/c_c_tip_how_detect_compiler_name_and_version_using_compiler_predefined_macros
constexpr bool is_clang =
#if defined(__clang__)
    true
#else
    false
#endif
    ;

constexpr bool is_gcc =
#if defined(__GNUC__) && !defined(__clang__)
    true
#else
    false
#endif
    ;

constexpr bool have_openblas =
#ifdef OPENBLAS_VERSION
    true
#else
    false
#endif
    ;

constexpr const char *openblas_version =
#ifdef OPENBLAS_VERSION
    OPENBLAS_VERSION
#else
    ""
#endif
    ;

constexpr const bool have_fast_math =
#ifdef __FAST_MATH__
    true
#else
    false
#endif
    ;
}  // namespace _build_info

inline std::string show_bool(bool b) { return b ? "true" : "false"; }

struct build_info_s {
    static const bool is_linux = _build_info::is_linux;
    static const bool is_mac = _build_info::is_mac;

    static const bool is_clang = _build_info::is_clang;
    static const bool is_gcc = _build_info::is_gcc;

    static const bool have_openblas = _build_info::have_openblas;
    const std::string openblas_version = _build_info::openblas_version;

    static const bool have_fast_math = _build_info::have_fast_math;

    std::string to_string()
    {
        constexpr const char *tab = "    ";
        std::stringstream ss;
        ss << "build info:\n"
           << "{\n";
        ss << tab << "os: " << (is_mac ? "mac" : "linux") << "\n";
        ss << tab << "compiler: " << (is_clang ? "clang" : "gcc") << "\n";
        ss << tab << "have_openblas: " << show_bool(have_openblas) << "\n";
        if (have_openblas) {
            ss << tab << "openblas version: " << openblas_version << "\n";
        }
        ss << tab << "have_fast_math: " << show_bool(have_fast_math) << "\n";
        ss << "}\n";
        return ss.str();
    }
};

template <typename F> class defer_t
{
    const F f_;

  public:
    explicit defer_t(const F &f) : f_(f) {}
    ~defer_t() { f_(); }
};

defer_t _show_build_info([] {
    build_info_s info;
    printf("%s\n", info.to_string().c_str());
});
