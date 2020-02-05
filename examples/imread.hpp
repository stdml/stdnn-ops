#pragma once
#include <ttl/tensor>

#include <opencv2/opencv.hpp>

namespace ttl::cv
{
::cv::Mat as_cv_mat(const ttl::tensor_ref<uint8_t, 3> &x)
{
    const auto [h, w, c] = x.dims();  // FIXME: check c==3
    if (c != 3) { throw std::invalid_argument(""); }
    return ::cv::Mat(::cv::Size(w, h), CV_8UC(3), x.data());
}

void imread_resize(const char *filename, const ttl::tensor_ref<uint8_t, 3> &x)
{
    const auto img = ::cv::imread(filename);
    auto resized_image = as_cv_mat(x);
    ::cv::resize(img, resized_image, resized_image.size(), 0, 0);
}
}  // namespace ttl::cv
