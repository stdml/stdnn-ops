#pragma once
namespace nn::ops
{
class noop
{
  public:
    template <typename T> void operator()(const T &y) const
    {
        // noop
    }

    template <typename S, typename T>
    void operator()(const T &y, const S &x) const
    {
        // noop
    }
};
}  // namespace nn::ops
