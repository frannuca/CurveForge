#pragma once
#include "pricing/curve/instrument.hpp"
#include <stdexcept>

namespace pricing {

class FRA : public CurveInstrument {
    double t1_, t2_, fwd_;
public:
    /**
     * Constructs a FRA (Forward Rate Agreement) instrument.
     * @param t1 Start time of the FRA period (in years).
     * @param t2 End time of the FRA period (in years).
     * @param fwd Forward rate agreed for the FRA period.
     */
    FRA(double t1, double t2, double fwd);
    double maturity() const override;
    const char* type() const override;
    double solveDiscount(const std::function<double(double)>& discount) const override;
};

} // namespace pricing
