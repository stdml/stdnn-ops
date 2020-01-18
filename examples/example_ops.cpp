#include <ttl/nn/ops>

void example_conv1d()
{
    ttl::nn::ops::conv1d f;
    ttl::tensor<float, 1> x(10);
    ttl::tensor<float, 1> y(3);
    ttl::tensor<float, 1> z(f(x.shape(), y.shape()));
    f(ref(z), view(x), view(y));
}

int main(int argc, char *argv[])
{
    example_conv1d();
    return 0;
}
