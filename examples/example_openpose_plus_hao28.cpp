// https://github.com/tensorlayer/openpose-plus/blob/master/openpose_plus/models/models_hao28_experimental.py

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include "example_openpose_plus_hao28.hpp"

using std::experimental::range;

int main(int argc, char *argv[])
{
    const std::string home(std::getenv("HOME"));
    const auto prefix = home + "/var/models/openpose";
    // const auto filename =
    //     home +
    //     "/var/data/openpose/examples/media/COCO_val2014_000000000192.jpg";

    openpose_plus_hao28 openpose(prefix);

    // auto paf_runner =
    //     create_paf_processor(32, 48, openpose.h, openpose.w, 19, 19, 13);

    auto x = ttl::tensor<float, 4>(1, openpose.h, openpose.w, 3);

    // TODO: input images
    // auto input = ttl::tensor<uint8_t, 4>(x.shape());
    // cv::Mat resized_image(cv::Size(openpose.w, openpose.h), CV_8UC(3),
    //                       input.data());
    // {
    //     auto img = cv::imread(filename);
    //     cv::resize(img, resized_image, resized_image.size(), 0, 0);
    //     std::transform(input.data(), data_end(input), x.data(),
    //                    [](uint8_t p) { return p / 255.0; });
    // }

    int repeats = 5;
    for (auto i : std::experimental::range(repeats)) {
        printf("inference %d\n", i);
        auto [l_conf, l_paf] = openpose(ref(x));

        // TODO: run paf process
        // auto conf = nn::ops::apply<ttl::tensor<float, 4>>(
        //     nn::ops::to_channels_first(), ref(*l_conf));
        // auto paf = nn::ops::apply<ttl::tensor<float, 4>>(
        //     nn::ops::to_channels_first(), ref(*l_paf));

        // auto human = (*paf_runner)(conf.data(), paf.data(), false);
        // for (auto h : human) {
        //     h.print();
        //     draw_human(resized_image, h);
        // }
        // cv::imwrite("a.png", resized_image);
    }
    return 0;
}
