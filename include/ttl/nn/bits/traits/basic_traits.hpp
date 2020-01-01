#pragma once

namespace nn
{
namespace traits
{
// image orders
struct nhwc;
struct nchw;
struct chw;
struct hwc;
struct hw;

// filter orders
struct rscd;
struct dcrs;

// for im2col output
struct hwrs;
struct hwrsc;
struct rshw;
}  // namespace traits
}  // namespace nn
