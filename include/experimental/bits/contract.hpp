#pragma once
#include <stdexcept>
#include <string>

// TODO: use c++20 contract
inline void contract_assert_(bool e, const char *file, int line)
{
    if (!e) {
        throw std::runtime_error(
            "contract_assert failed: " + std::string(file) + ":" +
            std::to_string(line));
    }
}

template <typename T>
inline void contract_assert_eq_(const T &x, const T &y, const char *file,
                                int line)
{
    if (x != y) {
        throw std::runtime_error(
            "contract_assert_eq failed: " + std::string(file) + ":" +
            std::to_string(line) + ": " + std::to_string(x) +
            " != " + std::to_string(y));
    }
}

#define contract_assert(e) contract_assert_((e), __FILE__, __LINE__)
#define contract_assert_eq(e, f)                                               \
    contract_assert_eq_((e), (f), __FILE__, __LINE__)
