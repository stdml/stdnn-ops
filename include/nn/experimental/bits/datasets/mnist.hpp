#pragma once
#include <string>

#include <nn/bits/ops/io.hpp>

namespace nn::experimental::datasets
{
struct mnist_data_set {
    const ttl::tensor<uint8_t, 3> images;
    const ttl::tensor<uint8_t, 1> labels;
};

// http://yann.lecun.com/exdb/mnist/
auto load_mnist_data(const std::string &data_dir, const std::string &name)
{
    const int n = [name] {
        if (name == "train") {
            return 60 * 1000;
        } else if (name == "t10k") {
            return 10 * 1000;
        } else {
            throw std::runtime_error("mnist name must be train or t10k");
        };
    }();
    ttl::tensor<uint8_t, 3> images(n, 28, 28);
    ttl::tensor<uint8_t, 1> labels(n);
    const std::string prefix = data_dir + "/" + name;
    nn::ops::readfile(prefix + "-images-idx3-ubyte")(ref(images));
    nn::ops::readfile(prefix + "-labels-idx1-ubyte")(ref(labels));
    return mnist_data_set{
        .images = std::move(images),
        .labels = std::move(labels),
    };
}
}  // namespace nn::experimental::datasets
