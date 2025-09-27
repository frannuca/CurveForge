#pragma once
#include "pricing/curve/instrument.hpp"
#include "pricing/curve/yield_curve.hpp"
#include <vector>

namespace pricing {

class CurveBootstrapper {
    std::vector<CurveInstrumentPtr> instruments_;
public:
    CurveBootstrapper& add(const CurveInstrumentPtr& inst);
    YieldCurve build() const;
};

} // namespace pricing
