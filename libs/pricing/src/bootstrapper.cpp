#include "pricing/curve/bootstrapper.hpp"
#include <algorithm>

namespace pricing {

CurveBootstrapper& CurveBootstrapper::add(const CurveInstrumentPtr& inst) {
    instruments_.push_back(inst);
    return *this;
}

YieldCurve CurveBootstrapper::build() const {
    std::vector<CurveInstrumentPtr> sorted = instruments_;
    std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b){ return a->maturity() < b->maturity(); });
    YieldCurve curve;
    auto discountAccessor = [&curve](double t){ return curve.discount(t); };
    for (auto &inst : sorted) {
        double df = inst->solveDiscount(discountAccessor);
        curve.add(inst->maturity(), df);
    }
    return curve;
}

} // namespace pricing

