#ifndef VSRP_H__
#define VSRP_H__ 
#include "blaze/Math.h"

namespace blaze {

//https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
//Very SparseRandomProjections

template<typename FT, bool SO=blaze::rowMajor, typename RNG>
CompressedMatrix<FT, SO> generate_vsrp(size_t nr, size_t nc, RNG &rng, double s=0.) {
    if(s == 0.) s = std::sqrt(nc);
    CompressedMatrix<FT, SO> ret(nr, nc);
    FT sqrts = std::sqrt(s);
    FT nsqrts = -sqrts;
    FT halfs = .5 * s;
    std::vector<unsigned> buf;
    buf.reserve(nc);
    std::uniform_real_distribution<FT> urd;
    ret.reserve((nr * nc * s) * 1.25);
    for(size_t i = 0; i < nr; ++i) {
        buf.clear();
        
        for(size_t j = 0; j < nc; ++j) {
            if(urd(rng) < halfs)
                ret.append(i, j, rng() % 2 ? sqrts: nsqrts);
        }
        for(const auto v: buf)
        ret.finalize(i);
    }
    return ret;
}

} // namespace blaze

#endif
