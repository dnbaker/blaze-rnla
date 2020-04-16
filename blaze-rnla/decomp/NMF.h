//=================================================================================================
/*!
//  \file blaze-rnla/decomp/NMF.h
//  \brief Header file for nonnegative matrix factorization
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

#ifndef _BLAZE_MATH_DENSE_NMF_H_
#define _BLAZE_MATH_DENSE_NMF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/functors/L2Norm.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/RemoveAdaptor.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/IsComplex.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/math/shims/Sqrt.h>
#include <blaze/util/Random.h>


#include <blaze-rnla/random/SVD.h>


namespace blaze {

namespace NMF {

/*
//
// NMF INITIALIZATION_STRATEGIES
//
//
//
 */

enum Initialization {
    NONE,       // Has been initialized already
    RANDOM,
    NNDSVD,
    NNDSVDA,
    NNDSVDAR,
    DEFAULT_INIT
};
/*
//
// NMF STRATEGIES
//
//
//
 */

enum Strategy {
    ALSQ, // Alternating Least Squares
    COORDINATE_DESCENT,
    MULTIPLICATIVE_UPDATE,
    CD=COORDINATE_DESCENT,
    MU=MULTIPLICATIVE_UPDATE
};

//=================================================================================================
//
//  NMF FUNCTIONS
//
//=================================================================================================


//*************************************************************************************************
/*!\name Eigenvalue functions */
//@{
//template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 >
//inline void nmf( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,TF>& w, DenseMatrix<MT3,SO2>& V );
//@}
//*************************************************************************************************


template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 , typename RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>>
inline auto nmf_alsq_backend(const Matrix<MT1,SO1>& A, DenseMatrix<MT2,TF>& H, DenseMatrix<MT3,SO2>& W, size_t ncomp, RT l2_reg=0., RT l1_reg=0., RT eps=1e-4)
 ->  EnableIf_t<IsFloatingPoint_v<ElementType_t<MT1>> && IsFloatingPoint_v<ElementType_t<MT2>> && IsFloatingPoint_v<ElementType_t<MT3>>>
{
    BLAZE_THROW_RUNTIME_ERROR("Not implemented");
}
template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 , typename RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>>
inline auto nmf_mu_backend(const Matrix<MT1,SO1>& A, DenseMatrix<MT2,TF>& H, DenseMatrix<MT3,SO2>& W, size_t ncomp, RT l2_reg=0., RT l1_reg=0., RT eps=1e-4)
 ->  EnableIf_t<IsFloatingPoint_v<ElementType_t<MT1>> && IsFloatingPoint_v<ElementType_t<MT2>> && IsFloatingPoint_v<ElementType_t<MT3>>>
{
    BLAZE_THROW_RUNTIME_ERROR("Not implemented");
}


template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 , typename RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>, typename Idx>
inline double nmf_cd_core(DenseMatrix<MT1,SO1>& W, const DenseMatrix<MT2,TF>& HHt, const Matrix<MT3,SO2>& XHt, const Idx &)
{
    BLAZE_FUNCTION_TRACE;
    RT viol = 0.;
    const DynamicVector<RT> invhess = map(band<0uL>(~HHt), [](auto x){return x ? 1. / x: 0.;});
    const size_t nsamp = (~W).rows(), ncomp = (~W).columns();
    for(unsigned i = 0; i < ncomp; ++i) {
        const auto comp = i ;//idx[i];
        auto hhrow = row(HHt, comp);
        RT local_viol = 0.;
#if BLAZE_OPENMP_PARALLEL_MODE
        _Pragma("omp parallel for reduction(+:local_viol)")
#endif
        for(size_t si = 0; si < nsamp; ++si) {
            auto wrow = row(W, si);
            ElementType_t<MT3> grad = -(~XHt)(si, comp) + sum(hhrow * wrow);
            auto &wr = wrow[comp];
            auto pg = wr ? grad: std::min(ElementType_t<MT3>(0.), grad);
            local_viol += std::abs(pg);
            wr = std::max(wr - invhess[comp] * grad, ElementType_t<MT3>(0.));
        }
        viol += local_viol;
    }
    return viol;
}

template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 , typename RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>, typename RNG, typename Idx>
inline double nmf_cd_update(const Matrix<MT1,SO1>& X, DenseMatrix<MT2,TF>& W, DenseMatrix<MT3,SO2>& Ht, RT l2_reg, RT l1_reg, RNG &rng, Idx &indices)
{
    BLAZE_FUNCTION_TRACE;
    auto XHt = evaluate((~X) * (~Ht));
    auto HHt = evaluate(trans(~Ht) * (~Ht));
    //std::shuffle(indices.begin(), indices.end(), rng);
    if(l2_reg > 0.)
        band<0uL>(~HHt) += l2_reg;
    if(l1_reg > 0.)
        (~XHt) -= l1_reg;
    return nmf_cd_core(W, HHt, XHt, indices);
}


template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 , typename RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>> >
inline auto nmf_cd_backend(const Matrix<MT1,SO1>& X, DenseMatrix<MT2,TF>& H, DenseMatrix<MT3,SO2>& W, size_t ncomp, RT l2_reg=0., RT l1_reg=0., RT eps=1e-4, size_t max_iter = 200, size_t seed = 0)
 ->  EnableIf_t<IsFloatingPoint_v<ElementType_t<MT1>> && IsFloatingPoint_v<ElementType_t<MT2>> && IsFloatingPoint_v<ElementType_t<MT3>>, size_t>
{
    MT2 Ht( trans(~H) );
    std::mt19937_64 mt(seed ? seed: size_t(std::time(nullptr)));
    double init_violation = 0.;
    size_t iternum;
    blaze::SmallArray<unsigned, 16> indicesHt((~Ht).columns()), indicesW((~W).columns());
    std::iota(indicesHt.begin(), indicesHt.end(), 0u);
    std::iota(indicesW.begin(), indicesW.end(), 0u);
    for( iternum = 0; ++iternum < max_iter; ) {
        double violation = nmf_cd_update(X, W, Ht, l2_reg, l1_reg, mt, indicesHt);
        violation += nmf_cd_update(trans(~X), Ht, W, l2_reg, l1_reg, mt, indicesW);
        if(iternum == 1) init_violation = violation;
        if(init_violation == 0.)              break;
        if(violation / init_violation <= eps) break;
    }
    if ( iternum == max_iter )
        std::fprintf(stderr, "Reached maximum # iterations: %zu\n", max_iter);
    std::fprintf(stderr, "Compressed matrix of %zu rows and %zu columns to H (%zu/%zu) and W (%zu/%zu) with %zu components in %zu iterations with eps = %g\n",
                 (~X).rows(), (~X).columns(), (~H).rows(), (~H).columns(), (~W).rows(), (~W).columns(), ncomp, iternum, eps);
    return iternum;
}

template< typename MT1, bool SO1, typename MT2, bool TF, typename MT3, bool SO2 , typename RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>> >
inline auto nmf_backend(const Matrix<MT1,SO1>& A, DenseMatrix<MT2,TF>& H, DenseMatrix<MT3,SO2>& W, size_t ncomp,
                        Strategy strategy=CD, RT l2_reg=0., RT l1_reg=0., RT eps=1e-4,
                        size_t max_iter = 200, size_t seed = 0, Initialization init=DEFAULT_INIT)
 ->  EnableIf_t<IsFloatingPoint_v<ElementType_t<MT1>> && IsFloatingPoint_v<ElementType_t<MT2>> && IsFloatingPoint_v<ElementType_t<MT3>>>
{
    // TODO: do NNDSVD initialization
    BLAZE_FUNCTION_TRACE;
    switch(strategy) {
        case CD:   nmf_cd_backend(A, H, W, ncomp, l2_reg, l1_reg, eps, max_iter, seed); break;
        case MU:   nmf_mu_backend(A, H, W, ncomp, l2_reg, l1_reg, eps); break;
        case ALSQ: nmf_alsq_backend(A, H, W, ncomp, l2_reg, l1_reg, eps); break;
        default: BLAZE_THROW_INVALID_ARGUMENT("Illegal strategy");
    }
}

template< typename MT1  // Type of the matrix A
        , bool SO1      // Storage order of the matrix A
        , typename MT2  // Type of the matrix H
        , bool SO2      // Storage order of the matrix H
        , typename MT3  // Type of the matrix V
        , bool SO3      // Storage order of the matrix V
        >
void init_nndsvd(const blaze::Matrix<MT1, SO1> &X, blaze::Matrix<MT2, SO2> &H, blaze::Matrix<MT3, SO3> &W, size_t ncomp, size_t seed, double eps=1e-5, bool avg=false, bool random=false) {
    BLAZE_FUNCTION_TRACE;
    using RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>;
    if(!IsDenseMatrix_v<MT1> || !IsDenseMatrix_v<MT2> || !IsDenseMatrix_v<MT3>) {
        BLAZE_THROW_RUNTIME_ERROR("All matrices must be dense for NNDSVD");
    }
    DynamicMatrix<RT> tmpU, tmpV;
    DynamicVector<RT> tmpS;
    std::fprintf(stderr, "initnndsvd \n");
    auto_svd(~X, tmpU, tmpS, tmpV, ncomp, (ncomp > 0 && ncomp < 5) ? 7: 6, 3, 1000);
    double svmul = sqrt(tmpS[0]);
    column(~W, 0) = svmul * abs(column(tmpU, 0));
    row(~H, 0)    = svmul * abs(row(tmpV, 0));
    for(size_t j = 1; j < ncomp; ++j) {
        auto x = column(tmpU, j);
        auto y = row(tmpV, j);
        const auto xp = max(x, RT(0)), yp = max(y, RT(0));
        const auto xn = abs(min(x, RT(0))), yn = abs(min(y, RT(0)));
        RT xpn = l2Norm(xp), ypn = l2Norm(yp), xnn = l2Norm(xn), ynn = l2Norm(yp);
        RT mp = xpn * ypn, mn = xnn * ynn;
        RT sigma;
        if(mp > mn) {
            sigma = mp;
            RT lbd = sqrt(sigma * tmpS[j]);
            column(~W, j) = (lbd / xpn) * xp;
            row(~H, j) = (lbd / ypn) * yp;
        } else {
            sigma = mn;
            RT lbd = sqrt(sigma * tmpS[j]);
            column(~W, j) = (lbd / xnn) * xn;
            row(~H, j) = (lbd / ynn) * yn;
        }
    }
    if(avg) { // nndsvda
        const ElementType_t<MT3> matmean = mean(~X);
        if(random) {
            std::mt19937_64 mt(seed);
            std::normal_distribution dist(matmean);
            auto epsormean = [&](auto x) {return x < eps ? dist(mt): x;};
            ~W = abs(map(~W, epsormean)) * .01;
            ~H = abs(map(~H, epsormean)) * .01;
        } else { // nndsvdar
            auto epsormean = [eps,matmean](auto x) {return x < eps ? matmean: x;};
            ~W = map(~W, epsormean);
            ~H = map(~H, epsormean);
        }
    } else {
        auto epsormore = [eps](auto x) {return x < eps ? ElementType_t<MT3>(0): x;};
        ~W = map(~W, epsormore);
        ~H = map(~H, epsormore);
    }
}

template< typename MT1  // Type of the matrix A
        , bool SO1      // Storage order of the matrix A
        , typename MT2  // Type of the matrix H
        , bool SO2      // Storage order of the matrix H
        , typename MT3  // Type of the matrix V
        , bool SO3      // Storage order of the matrix V
        >
void init_nndsvda(const blaze::Matrix<MT1, SO1> &X, blaze::Matrix<MT2, SO2> &H, blaze::Matrix<MT3, SO3> &W, size_t ncomp, size_t seed, double eps=1e-5) {
    init_nndsvd(X, H, W, ncomp, seed, true, false);
}

template< typename MT1  // Type of the matrix A
        , bool SO1      // Storage order of the matrix A
        , typename MT2  // Type of the matrix H
        , bool SO2      // Storage order of the matrix H
        , typename MT3  // Type of the matrix V
        , bool SO3      // Storage order of the matrix V
        >
void init_nndsvdar(const blaze::Matrix<MT1, SO1> &X, blaze::Matrix<MT2, SO2> &H, blaze::Matrix<MT3, SO3> &W, size_t ncomp, size_t seed, double eps=1e-5) {
    init_nndsvd(X, H, W, ncomp, seed, true, true);
}

template< typename MT1  // Type of the matrix A
        , bool SO1      // Storage order of the matrix A
        , typename MT2  // Type of the matrix H
        , bool SO2      // Storage order of the matrix H
        , typename MT3  // Type of the matrix V
        , bool SO3      // Storage order of the matrix V
        >
void init_random(const blaze::Matrix<MT1, SO1> &X, blaze::Matrix<MT2, SO2> &H, blaze::Matrix<MT3, SO3> &W, size_t ncomp, size_t seed, double eps=1e-5) {
    using RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>;
    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<RT> dist(0., sqrt(mean(~X) / ncomp));
    auto func = [&](size_t,size_t) {
        return dist(mt);
    };
    ~H = generate((~H).rows(), (~H).columns(), func); ~H = abs(~H);
    ~W = generate((~W).rows(), (~W).columns(), func); ~W = abs(~W);
}

template< typename MT1  // Type of the matrix A
        , bool SO1      // Storage order of the matrix A
        , typename MT2  // Type of the matrix H
        , bool SO2      // Storage order of the matrix H
        , typename MT3  // Type of the matrix V
        , bool SO3      // Storage order of the matrix V
        >
inline auto nmf_initialize(const blaze::Matrix<MT1, SO1> &X, blaze::Matrix<MT2, SO2> &H, blaze::Matrix<MT3, SO3> &W,
                           size_t ncomp, Initialization init, size_t seed, double eps)
{
    BLAZE_FUNCTION_TRACE;
    using RT = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>;
    if((~H).rows() != ncomp || (~H).columns() != (~X).columns() ) {
        (~H).resize(ncomp, (~X).columns());
    }
    if((~W).columns() != ncomp || (~W).rows() != (~X).rows() ) {
        (~W).resize((~X).rows(), ncomp);
    }
    const size_t nsamp = (~X).rows();
    bool a = false, r = false;
    switch(init) {
        case RANDOM:
            init_random(X, H, W, ncomp, seed, eps); break;
        case NONE:
            break;
        case DEFAULT_INIT:
            if(ncomp <= std::min((~X).rows(), (~X).columns()))
                init_nndsvd(X, H, W, ncomp, seed, eps);
            else
                init_random(X, H, W, ncomp, seed, eps); break;
        case NNDSVDAR:
            r = true;
            [[fallthrough]];
        case NNDSVDA:
            a = true;
            [[fallthrough]];
        case NNDSVD:
            init_nndsvd(X, H, W, ncomp, seed, eps, a, r); break;
        default: BLAZE_THROW_INVALID_ARGUMENT("Unknown initialization");
    }
}

//*************************************************************************************************
/*!\brief Nonnegative Matrix Factorization of a given matrix
// \ingroup matrix
//
// \param A The given general matrix.
// \param U The resulting matrix of left support vectors.
// \param S The resulting vector of singular values.
    A general matrix can be multiplied by it as result = inmat % blaze::expand(trans(S), (~A).rows())
// \param V The resulting matrix of left support vectors.
// \param ncomp The number of components to return. By default, the decomposition is complete.
// \param All further parameters control the nature of regularization and optimization. See code for details.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
// \exception std::invalid_argument Vector cannot be resized.
// \exception std::invalid_argument Matrix cannot be resized.
// \exception std::runtime_error Eigenvalue computation failed.
//
// This function computes the eigenvalues and eigenvectors of the given \a n-by-\a n matrix.
// The eigenvalues are returned in the given vector \a w and the eigenvectors are returned in the
// given matrix \a V, which are both resized to the correct dimensions (if possible and necessary).
//
// Please note that in case the given matrix is either a compile time symmetric matrix with
// floating point elements or an Hermitian matrix with complex elements, the resulting eigenvalues
// will be of floating point type and therefore the elements of the given eigenvalue vector are
// expected to be of floating point type. In all other cases they are expected to be of complex
// type. Also please note that for complex eigenvalues no order of eigenvalues can be assumed,
// except that complex conjugate pairs of eigenvalues appear consecutively with the eigenvalue
// having the positive imaginary part first.
//
// In case \a A is a row-major matrix, \a V will contain the left eigenvectors, otherwise \a V
// will contain the right eigenvectors. In case \a V is a row-major matrix the eigenvectors are
// returned in the rows of \a V, in case \a V is a column-major matrix the eigenvectors are
// returned in the columns of \a V. In case the given matrix is a compile time symmetric matrix
// with floating point elements, the resulting eigenvectors will be of floating point type and
// therefore the elements of the given eigenvector matrix are expected to be of floating point
// type. In all other cases they are expected to be of complex type.
//
// The function fails if ...
//
//  - ... the given matrix \a A is not a square matrix;
//  - ... the given vector \a w is a fixed size vector and the size doesn't match;
//  - ... the given matrix \a V is a fixed size matrix and the dimensions don't match;
//  - ... the eigenvalue computation fails.
//
// In all failure cases an exception is thrown.
//
// Examples:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   DynamicMatrix<double,rowMajor> A( 5UL, 5UL );  // The general matrix A
   // ... Initialization

   DynamicVector<complex<double>,columnVector> w( 5UL );   // The vector for the complex eigenvalues
   DynamicMatrix<complex<double>,rowMajor> V( 5UL, 5UL );  // The matrix for the left eigenvectors

   eigen( A, w, V );
   \endcode

   \code
   using blaze::SymmetricMatrix;
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   SymmetricMatrix< DynamicMatrix<double,rowMajor> > A( 5UL );  // The symmetric matrix A
   // ... Initialization

   DynamicVector<double,columnVector> w( 5UL );       // The vector for the real eigenvalues
   DynamicMatrix<double,rowMajor>     V( 5UL, 5UL );  // The matrix for the left eigenvectors

   eigen( A, w, V );
   \endcode

   \code
   using blaze::HermitianMatrix;
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   HermitianMatrix< DynamicMatrix<complex<double>,rowMajor> > A( 5UL );  // The Hermitian matrix A
   // ... Initialization

   DynamicVector<double,columnVector>      w( 5UL );       // The vector for the real eigenvalues
   DynamicMatrix<complex<double>,rowMajor> V( 5UL, 5UL );  // The matrix for the left eigenvectors

   eigen( A, w, V );
   \endcode

// \note This function only works for matrices with \c float, \c double, \c complex<float>, or
// \c complex<double> element type. The attempt to call the function with matrices of any other
// element type results in a compile time error!
//
// \note This function can only be used if a fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
//
// \note Further options for computing eigenvalues and eigenvectors are available via the geev(),
// syev(), syevd(), syevx(), heev(), heevd(), and heevx() functions.
*/
template< typename MT1  // Type of the matrix A
        , bool SO1      // Storage order of the matrix A
        , typename MT2  // Type of the matrix H
        , bool SO2      // Storage order of the matrix H
        , typename MT3  // Type of the matrix V
        , bool SO3      // Storage order of the matrix V
        , typename RT   // RegularizationType: CommonType of all elements
          = CommonType_t<ElementType_t<MT1>, ElementType_t<MT2>, ElementType_t<MT3>>
        >
inline void nmf( const Matrix<MT1,SO1>& A,
                 DenseMatrix<MT2,SO2>& H,
                 DenseMatrix<MT3,SO3>& W,
                 size_t ncomp,
                 Strategy strategy   = COORDINATE_DESCENT,
                 RT l2_reg           = 0.,
                 RT l1_reg           = 0.,
                 RT eps              = 1e-4,
                 size_t max_iter     = 10000,
                 Initialization init = DEFAULT_INIT,
                 size_t seed         = 0)
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT2 );

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT3 );
   if ( ! ncomp )
       ncomp = (~A).columns();

   using WTmp = If_t< IsContiguous_v<MT2>, MT2&, ResultType_t<MT2> >;
   using MT2mp = If_t< IsContiguous_v<MT3>, MT3&, ResultType_t<MT3> >;


   WTmp Htmp( ~H );
   MT2mp Wtmp( ~W );
   nmf_initialize(A, Htmp, Wtmp, ncomp, init, seed, eps);

   nmf_backend( ~A, Htmp, Wtmp, ncomp, strategy, l2_reg, l1_reg, eps, max_iter, seed);

   if( !IsContiguous_v<MT2> ) {
      (~H) = Htmp;
   }

   if( !IsContiguous_v<MT3> ) {
      (~W) = Wtmp;
   }
}
//*************************************************************************************************

} // namespace NMF
using NMF::nmf;
} // namespace blaze

#endif
