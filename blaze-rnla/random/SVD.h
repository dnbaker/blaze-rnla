//=================================================================================================
/*!
//  \file blaze/math/random/SVD.h
//  \brief Header file for randomized Singular Value Decomposition
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

#ifndef _BLAZE_MATH_RANDOM_SVD_H_
#define _BLAZE_MATH_RANDOM_SVD_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/dense/DenseVector.h>
#include <blaze/math/dense/DenseMatrix.h>
#include <blaze/math/dense/SVD.h>
#include <blaze/Math.h>

#include <blaze-rnla/config/Randomized.h>
#include <blaze-rnla/random/RangeFinder.h>

namespace blaze {

namespace detail {

template<typename MT, bool SO>
void gram_schmidt(blaze::DenseMatrix<MT, SO> &mat, const ElementType_t<MT> EPS=1e-6) {
    using Scalar = ElementType_t<MT>;
    auto &mr = ~mat;
    for(size_t i = 0; i < mr.rows(); ++i)
    {
        auto mri = row((~mat), i);
        for(size_t j = 0; j < i; ++j)
        {
            auto mrj = row((~mat), j);
            const Scalar r = dot(mrj, mri);
            mri -= r * mrj;
        }
        const Scalar norm = l2Norm(mri);
        if(norm < EPS)
        {
            blaze::submatrix(mr, i, 0, mr.rows() - i, mr.columns()) = Scalar(0);
            return;
        }
        mri /= norm;
    }
}

} // detail

template<typename MT, bool SO, typename MT2, typename VT, bool TF, typename MT3>
void randomized_svd(const Matrix<MT, SO> &matrix, DenseMatrix<MT2, SO> &U,  DenseVector<VT, TF> &S, DenseMatrix<MT3, SO> &V,
                    size_t l=0, unsigned niter=4)
{
    using ET = ElementType_t<MT>;
    // First try: Straight from Tropp
    using std::swap;
    auto q = evaluate(find_range_approx(matrix, l, niter));
    DynamicMatrix<ET, SO> B = ctrans(q) * ~matrix;
    svd(B, ~U, ~S, ~V);
    ResultType_t<MT2> Up = q * trans(~U);
    swap(Up, ~U);
}

template<typename MT, bool SO>
void randomized_svd(const Matrix<MT, SO> &matrix, size_t l=0, unsigned niter=4) {
    using ET = ElementType_t<MT>;
    blaze::DynamicMatrix<ET, SO> U, V;
    blaze::DynamicVector<ET> S;
    randomized_svd(matrix, U, S, V, l, niter);
}


// Automatically selects randomized or default SVD
template<typename MT, bool SO, typename MT2, typename VT, bool TF, typename MT3, typename=EnableIf_t<IsDenseMatrix_v<MT>> >
void auto_svd(const DenseMatrix<MT, SO> &matrix, DenseMatrix<MT2, SO> &U,  DenseVector<VT, TF> &S, DenseMatrix<MT3, SO> &V,
              size_t l=0, unsigned niter=4,
              unsigned oversampling=3, const size_t threshold=BLAZE_RANDOMIZED_SVD_THRESHOLD)
{
    // TODO: specialize this for un-resizable matrices for U, V
    const size_t mn = std::min((~matrix).rows(), (~matrix).columns()), mx = std::max((~matrix).rows(), (~matrix).columns());
    if(l && l + oversampling <= mn && mx * l >= threshold) {
        const size_t sampled = std::min(l + oversampling, mn);
        randomized_svd(matrix, U, S, V, sampled, niter);
        if(sampled != l) {
            ~S = subvector(~S, 0, l);
            ~U = submatrix(~U, 0, 0, (~U).rows(), l);
            ~V = submatrix(~V, 0, 0, l, (~V).columns());
        }
    } else {
        if(l)
            svd(matrix, U, S, V, size_t(0), l - 1);
        else
            svd(matrix, U, S, V);
    }
}

// Automatically selects randomized or default SVD
template<typename MT, bool SO, typename MT2, typename VT, bool TF, typename MT3, typename=EnableIf_t<IsSparseMatrix_v<MT>> >
void auto_svd(const SparseMatrix<MT, SO> &matrix, DenseMatrix<MT2, SO> &U,  DenseVector<VT, TF> &S, DenseMatrix<MT3, SO> &V,
              size_t l=0, unsigned niter=4,
              unsigned oversampling=3, const size_t threshold=BLAZE_RANDOMIZED_SVD_THRESHOLD)
{
    // TODO: specialize this for un-resizable matrices for U, V
    const size_t mn = std::min((~matrix).rows(), (~matrix).columns()), mx = std::max((~matrix).rows(), (~matrix).columns());
    if(!l) l = mn;
    const size_t sampled = std::min(l + oversampling, mn);
    randomized_svd(matrix, U, S, V, sampled, niter);
    if(sampled != l) {
        ~S = subvector(~S, 0, l);
        ~U = submatrix(~U, 0, 0, (~U).rows(), l);
        ~V = submatrix(~V, 0, 0, l, (~V).columns());
    }
}


} // namespace blaze

#endif
