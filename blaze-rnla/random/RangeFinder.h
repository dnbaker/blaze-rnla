//=================================================================================================
/*!
//  \file blaze-rnla/random/RangeFinder.h
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

#ifndef _BLAZE_MATH_RANDOM_RANGEFINDER_H_
#define _BLAZE_MATH_RANDOM_RANGEFINDER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/PaddingFlag.h>
#include <blaze/math/dense/QR.h>
#include <blaze/math/dense/LU.h>
#include <blaze/system/Alignment.h>
#include <blaze/system/Padding.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/RemoveConst.h>
#include <blaze/util/Random.h>
#include <blaze/Math.h>
#include <blaze/util/typetraits/IsComplex.h>

#include <blaze-rnla/random/GaussianMatrix.h>

namespace blaze {

template<typename MT, bool SO, typename MT2, bool SO2, typename MT3, bool SO3>
void find_range_approx_simple(const Matrix<MT, SO> &matrix, Matrix<MT2, SO2> &q, Matrix<MT3, SO3> &r, const size_t l)
{
    using ET = ElementType_t<MT>;
    const size_t nc = (~matrix).columns(), nr = (~matrix).rows();
    const auto gaussian_matrix = evaluate(make_gaussian_matrix<ET>(nc, l));
    auto prod = evaluate((~matrix) * gaussian_matrix);
    qr(prod, ~q, ~r);
}

template<typename MT, bool SO>
decltype(auto) find_range_approx_simple(const Matrix<MT, SO> &matrix, const size_t l)
{
    using ET = ElementType_t<MT>;
    blaze::DynamicMatrix<ET> q, r;
    find_range_approx_simple(matrix, q, r, l);
    auto qnewmat = q * ctrans(q) * (~matrix) - ~matrix;
    return q;
}


template<typename MT, bool SO>
decltype(auto) find_range_approx_rsi(const Matrix<MT, SO> &matrix, const size_t l, unsigned niter)
{
    // Randomized Subspace Iteration
    using ET = ElementType_t<MT>;
    const size_t nc = (~matrix).columns(), nr = (~matrix).rows();
    DynamicMatrix<ET> q, r, tmp;
#ifndef NDEBUG
    std::fprintf(stderr, "First range\n");
#endif
    find_range_approx_simple(matrix, q, r, l);
    for(unsigned i = 0; i < niter; ++i) {
#ifndef NDEBUG
        std::fprintf(stderr, "%zuth range, first matmul\n", i);
#endif
        tmp = ctrans(~matrix) * q;
#ifndef NDEBUG
        std::fprintf(stderr, "%zuth range, first QR\n", i);
#endif
        qr(tmp, q, r);
#ifndef NDEBUG
        std::fprintf(stderr, "%zuth range, second matmul\n", i);
#endif
        tmp = (~matrix) * q;
#ifndef NDEBUG
        std::fprintf(stderr, "%zuth range, second QR\n", i);
#endif
        qr(tmp, q, r);
    }
    return q;
}

template<typename MT, bool SO>
decltype(auto) find_range_approx(const Matrix<MT, SO> &matrix, size_t l, unsigned niter=0)
{
    if(l == 0) l = std::min((~matrix).rows(), (~matrix).columns());
    if(!niter) {
        return find_range_approx_simple(matrix, l);
    }
    // TODO: Implement using subsampled FFT construction.
    // This will require linking against an FFT implementation.
    return find_range_approx_rsi(matrix, l, niter);
}

} // namespace blaze

#endif
