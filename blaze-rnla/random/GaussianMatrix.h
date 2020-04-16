//=================================================================================================
/*!
//  \file blaze/math/random/GaussianMatrix.h
//  \brief Header file for all forward declarations for random vectors, matrices,
//         and randomized algorithms
//
//  Copyright (C) 2012-2020 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_RANDOM_GAUSSIANMATRIX_H_
#define _BLAZE_MATH_RANDOM_GAUSSIANMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/functors/Abs.h>
#include <blaze/math/PaddingFlag.h>
#include <blaze/math/expressions/DMatGenExpr.h>
#include <blaze/system/Alignment.h>
#include <blaze/system/Padding.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/Random.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/RemoveConst.h>
#include <blaze/util/constraints/FloatingPoint.h>
#include <thread>


namespace blaze {

namespace detail {

static inline constexpr uint64_t _wymum(uint64_t x, uint64_t y)
{
    __uint128_t l = __uint128_t(x) * y;
    return l ^ (l >> 64);
}

static inline constexpr uint64_t wyhash64_stateless(uint64_t *seed)
{
    *seed += UINT64_C(0x60bee2bee120fc15);
    return _wymum(*seed ^ 0xe7037ed1a0b428dbull, *seed);
}

} // namespace detail

template<typename FT=double, bool SO=rowMajor>
decltype(auto) make_gaussian_matrix(size_t nrows, size_t ncol, size_t seed = 0) {
    BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE(FT);
    std::fprintf(stderr, "making gaussian matrix\n");
    if(seed == 0) {
        randomize(seed);
    }
    return generate<SO>(nrows, ncol, [&](uint64_t x, uint64_t y) {
        std::normal_distribution<FT> dist(0, !IsComplex_v<FT> ? 1.: 0.707106781186547);
        x ^= seed;
        detail::wyhash64_stateless(&x);
        y = detail::_wymum(y ^ x, y);
        DefaultRNG rng(y);
        return dist(rng);
    });
}

template<typename FT=double, bool SO=rowMajor>
decltype(auto) make_nonnegative_gaussian_matrix(size_t nrows, size_t ncol, size_t seed = 0)
{
    return abs(make_gaussian_matrix<FT, SO>(nrows, ncol, seed));
}

} // namespace blaze

#endif
