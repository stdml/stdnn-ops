#include <cstdlib>

#include <iostream>

#include <gtest/gtest.h>
#include <ttl/filesystem>

int loc(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");
    if (fp == nullptr) { return 0; }
    constexpr int max_line = 1 << 16;
    char line[max_line];
    int ln = 0;
    while (std::fgets(line, max_line - 1, fp)) { ++ln; }
    std::fclose(fp);
    return ln;
}

void test_dir_loc(const char *path, const int file_limit, const int tot_limit,
                  int &acc)
{
    namespace fs = std::filesystem;

    int tot = 0;
    int n = 0;
    for (const auto &entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry)) {
            const int ln = loc(entry.path().c_str());
            printf("%4d %s\n", ln, entry.path().c_str());
            ASSERT_TRUE(ln <= file_limit);
            tot += ln;
            ++n;
        }
    }
    printf("total: %d lines in %d files\n", tot, n);
    ASSERT_TRUE(tot <= tot_limit);
    acc += tot;
}

TEST(test_loc, test1)
{
    int acc = 0;
    test_dir_loc("include/ttl/nn/bits/kernels", 60, 600, acc);
    test_dir_loc("include/ttl/nn/bits/kernels/cpu", 200, 2500, acc);
    test_dir_loc("include/ttl/nn/bits/ops", 300, 2500, acc);
    test_dir_loc("include/ttl/nn/bits/ops/gradients", 300, 2500, acc);
    printf("total %d lines", acc);
    ASSERT_TRUE(acc <= 6000);
}
