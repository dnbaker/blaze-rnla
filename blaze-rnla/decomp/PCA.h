#ifndef BLAZE_DECOMPOSITION_PCA_H__
#define BLAZE_DECOMPOSITION_PCA_H__

/*
 * blaze-rnla/math/decomp/PCA.h
 * Contains the Blaze headers for matrix decompsition.
 */
#include "blaze-rnla/decomp/NMF.h"

namespace blaze {

namespace PCA {

template<typename FT, bool TF>
class CovarianceAccumulator {
    DynamicVector<FT, TF> accumulation_;
    DynamicVector<FT, TF> means_;
    size_t n_added_;
    const size_t dim_;
    static size_t dim2nelem(size_t n) {
        return (n * (n + 1)) / 2;
    }
public:
    CovarianceAccumulator(size_t dim): accumulation_(dim2nelem(dim)), means_(dim), n_added_(0), dim_(dim) {}
    template<typename VT>
    void add(const DenseVector<VT, TF> &v) {
        auto &vr = ~v;
        BLAZE_INTERNAL_ASSERT(vr.size() == dim_, "Incorrect vector size detected" );
        if(++n_added_ == 1) { // Use Welford's running sd algorithm for better accuracy
            means_ = vr;
        } else {
            means_ += (vr - means_) / n_added_;
        }
        size_t subv_length = dim_, offset = 0;
        for(size_t i = 0; i < dim_; ++i) {
            auto subv = subvector(accumulation_, offset, subv_length);
            auto vsubv = subvector(vr, i, subv_length);
            subv += vsubv[0] * vsubv;
            offset += subv_length--;
        }
    }
    template<typename VT>
    void add(const SparseVector<VT, TF> &v) {
        auto &vr = ~v;
        BLAZE_INTERNAL_ASSERT(vr.size() == dim_, "Incorrect vector size detected" );
        if(++n_added_ == 1) { // Use Welford's running sd algorithm for better accuracy
            means_ = vr;
        } else {
            means_ += (vr - means_) / n_added_;
        }
        for(const auto &pair: v) {
            auto val = pair.value();
            auto index = pair.index();
            size_t offset = dim_ * index - ((index - 1) * index);
            const size_t subv_len = dim_ - index;
            subvector(accumulation_, offset, subv_len) += subvector(vr, index, subv_len);
            // Offset into array where the subvector starts
        }
    }
    template<typename VT>
    void add(const Vector<VT, !TF> &v) {
        add(trans(~v));
    }
    template<ReductionFlag RF=rowwise, typename MT, bool SO>
    void add(const Matrix<MT, SO> &m) {
        auto &mr = (~m);
        if(RF == rowwise) {
            BLAZE_INTERNAL_ASSERT(mr.columns() == dim_, "Incorrect matrix dimensions detected.");
            const size_t nr = mr.rows();
            for(size_t i = 0; i < nr; ++i) {
                add(row(mr, i, unchecked));
            }
        } else {
            BLAZE_INTERNAL_ASSERT(mr.rows() == dim_, "Incorrect matrix dimensions detected.");
            const size_t nc = mr.columns();
            for(size_t i = 0; i < nc; ++i) {
                add(column(mr, i, unchecked));
            }
        }
    }
    DynamicVector<FT, TF> covariance_array() const {
        DynamicVector<FT, TF> ret = accumulation_;
        size_t offset = 0, subv_length = dim_;
        ret *= FT(1. / n_added_);
        for(size_t i = 0; i < dim_; ++i) {
            auto subv = subvector(ret, offset, subv_length);
            subv -= means_[i] * subvector(means_, i, subv_length);
            offset += subv_length--;
        }
        return ret;
    }
    template<bool SO=rowMajor>
    decltype(auto) covariance_matrix() const {
        auto ca = covariance_array();
        DynamicMatrix<FT, SO> ret(dim_, dim_);
        size_t offset = 0, subv_length = dim_;
        for(size_t i = 0; i < dim_; ++i) {
            subvector(row(ret, i, unchecked), i, subv_length) =
            trans(subvector(ca, offset, subv_length));
            subvector(column(ret, i, unchecked), i, subv_length) = subvector(ca, offset, subv_length);
            offset += subv_length--;
        }
        return ret;
    }
    template<bool SO=rowMajor>
    std::pair<DynamicMatrix<FT, SO>, DynamicMatrix<FT, SO>>
    compute_pca(size_t l, unsigned oversampling=4) const {
        auto cm = evaluate(declsym(covariance_matrix<SO>()));
        DynamicMatrix<FT, SO> U, V;
        DynamicVector<FT> S;
        auto_svd(cm, U, S, V, l, oversampling);
        DynamicMatrix<FT, SO> scores = trans(expand(S, dim_)) % U;
        return std::make_pair(V, scores);
    }
};

template<typename MT, bool SO>
auto pca(const Matrix<MT, SO> &mat, unsigned ncomp=0, bool byrow=true) {
    ResultType_t<MT> tmpM(~mat);
    CovarianceAccumulator<ElementType_t<MT>, SO> ca(byrow ? tmpM.columns(): tmpM.rows());
    if(byrow)
        ca.add<rowwise>(tmpM);
    else
        ca.add<columnwise>(tmpM);
    if(!ncomp) ncomp = byrow ? tmpM.columns(): tmpM.rows();
    return ca.compute_pca(ncomp);
}

} // PCA
using PCA::CovarianceAccumulator;

}


#endif /* BLAZE_DECOMPOSITION_PCA_H__ */
