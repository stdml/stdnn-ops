#pragma once

namespace std::experimental
{

/*! new_type implements the newtype keyword from Haskell */
template <typename T, typename K> class new_type : public T
{
    using T::T;
};

}  // namespace std::experimental
