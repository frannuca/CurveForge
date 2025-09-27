#pragma once
#include "pricing/curve/instrument.hpp"
#include <stdexcept>

namespace pricing {

class OISDeposit : public CurveInstrument {
    double T_; double rate_;
public:
    OISDeposit(double T, double rate);
    double maturity() const override;
    const char* type() const override;
    double solveDiscount(const std::function<double(double)>&) const override;
};

} // namespace pricing
