#pragma once
#include <stdexcept>
#include <string>

// TODO: use c++20 contract
inline void contract_assert_(bool e, const char *file, int line)
{
    if (!e) {
        // TODO: add source location
        throw std::runtime_error(
            "contract_assert failed: " + std::string(file) + ":" +
            std::to_string(line));
    }
}

#define contract_assert(e) contract_assert_((e), __FILE__, __LINE__)
