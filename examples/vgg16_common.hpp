#pragma once
#include <string>
#include <vector>

#include <ttl/tensor>

#ifdef USE_OPENCV
#    include "imread.hpp"
#endif

std::vector<std::string> load_class_names(const std::string &filename)
{
    std::vector<std::string> names;
    std::string line;
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("file not found: " + filename);
    }
    while (std::getline(in, line)) { names.push_back(line); }
    return names;
}

void read_example_image(const ttl::tensor_ref<float, 4> &x)
{
#ifdef USE_OPENCV
    if (const int code =
            system("[ ! -f laska.png ] && curl -vLOJ "
                   "https://www.cs.toronto.edu/~frossard/vgg16/laska.png");
        code != 0) { /* noop */
    }
    auto input = ttl::tensor<uint8_t, 4>(x.shape());
    ttl::cv::imread_resize("laska.png", input[0]);
    for (auto i : ttl::range<1>(x)) {
        for (auto j : ttl::range<2>(x)) {
            x.at(0, i, j, 0) = input.at(0, i, j, 2);
            x.at(0, i, j, 1) = input.at(0, i, j, 1);
            x.at(0, i, j, 2) = input.at(0, i, j, 0);
        }
    }
#else
    (ttl::nn::ops::readfile(prefix + "/laska.idx"))(x[0]);
#endif
    std::vector<float> mean({123.68, 116.779, 103.939});
    ttl::nn::ops::apply_bias<ttl::nn::ops::nhwc, std::minus<float>>()(
        x, ttl::view(x), ttl::tensor_view<float, 1>(mean.data(), mean.size()));
}
