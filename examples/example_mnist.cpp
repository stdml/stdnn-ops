#include <cstdlib>

#include <string>

#include <nn/ops>

#ifdef USE_OPENCV
#    include <opencv2/opencv.hpp>
#endif

#include <ttl/tensor>

#include "utils.hpp"

void example_mnist() {}

int main()
{
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";

    std::string filename = prefix + "/" + "t10k-images-idx3-ubyte";

    using reader = nn::ops::readfile;
    ttl::tensor<uint8_t, 3> t(10 * 1000, 28, 28);
    (reader(filename))(ref(t));

    int i = 0;
    int code = system("mkdir -p images");
    UNUSED(code);
    for (auto im : t) {
        UNUSED(im);
        char name[32];
        sprintf(name, "images/%d.png", ++i);
#ifdef USE_OPENCV
        cv::Mat img(cv::Size(28, 28), CV_8UC(1), im.data());
        cv::imwrite(name, img);
#endif
        // if (i >= 100) { break; }
    }
    return 0;
}
