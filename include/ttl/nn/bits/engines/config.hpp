#pragma once
#include <ttl/nn/bits/engines/builtin.hpp>
#include <ttl/nn/bits/engines/plain_impl.hpp>
#ifdef STDNN_OPS_HAVE_CBLAS
#    include <ttl/nn/bits/engines/cblas_impl.hpp>
#endif

namespace ttl::nn::engines
{
struct cblas;

template <typename engine>
struct backend;
template <>
struct backend<builtin> {
    template <typename R>
    using type = plain_impl<R>;
};

#ifdef STDNN_OPS_HAVE_CBLAS

using default_engine = cblas;

template <>
struct backend<cblas> {
    template <typename R>
    using type = cblas_impl<R>;
};

#else

using default_engine = builtin;

#endif
}  // namespace ttl::nn::engines
