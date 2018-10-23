#pragma once
#include <nn/common.hpp>

#include <nn/bits/engines/plain_impl.hpp>

#ifdef STDNN_OPS_HAVE_CBLAS
#include <nn/bits/engines/cblas_impl.hpp>

namespace nn::engines
{
template <typename T> using default_engine = cblas_impl<T>;
}

#else

namespace nn::engines
{
template <typename T> using default_engine = plain_impl<T>;
}

#endif
