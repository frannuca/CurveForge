#include "pricing/engine.hpp"
#include "mathx/mathx.hpp"
#include <stdexcept>

namespace pricing {
    double par_rate_example(const std::vector<double>& cfs,
                            const std::vector<double>& D) {
        if (cfs.size() != D.size()) throw std::invalid_argument("size mismatch");
        double pv = 0.0, ann = 0.0;
        for (size_t i=0;i<cfs.size();++i){ pv += cfs[i]*D[i]; ann += D[i]; }
        if (ann == 0.0) return 0.0;
        return pv/ann; // toy formula
    }
}