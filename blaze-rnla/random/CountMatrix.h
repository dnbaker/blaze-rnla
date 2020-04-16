//=================================================================================================
/*!
//  \file blaze-rnla/random/CountMatrix.h
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

#ifndef _BLAZE_MATH_RANDOM_COUNTMATRIX_H_
#define _BLAZE_MATH_RANDOM_COUNTMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/PaddingFlag.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/DenseMatrix.h>
#include <blaze/system/Alignment.h>
#include <blaze/system/Padding.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/Random.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/RemoveConst.h>
#include <blaze/util/typetraits/IsNumeric.h>

namespace blaze {

template<typename FT=double, bool SO=rowMajor, typename RNG=DefaultRNG, typename=EnableIf_t<IsNumeric_v<FT>>>
blaze::CompressedMatrix<FT, SO>
make_count_matrix(size_t nrows, size_t ncol, size_t seed = 0) {
    if(seed) {
        Random<RNG>::rng_.seed(seed);
    }
    blaze::CompressedMatrix<FT, SO> ret(nrows, ncol);
    ret.reserve(nrows);
    for(size_t i = 0; i < nrows; ++i) {
        auto gv =  Random<RNG>::rng_();
        ret.append(i, gv % ncol, gv >> (sizeof(gv) * 8 - 1) ? FT(-1): FT(1));
        ret.finalize(i);
    }
    return ret;
}

template<typename FT=double, bool SO=rowMajor, typename RNG=DefaultRNG, typename=EnableIf_t<IsNumeric_v<FT>>>
blaze::DenseMatrix<FT, SO>
make_dense_count_matrix(size_t nrows, size_t ncol, size_t seed = 0) {
    if(seed) {
        Random<RNG>::rng_.seed(seed);
    }
    blaze::DenseMatrix<FT, SO> ret(nrows, ncol, FT(0));
    for(size_t i = 0; i < nrows; ++i) {
        auto gv = Random<RNG>::rng_();
        ret(i, gv % ncol) = gv >> (sizeof(gv) * 8 - 1) ? FT(-1): FT(1);
    }
}


} // namespace blaze

#endif
