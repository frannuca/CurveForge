#pragma once
#include "pricing/curve/instrument.hpp"
#include <stdexcept>

namespace pricing {

class Futures : public CurveInstrument {
    double t1_, t2_, implied_;
public:
    Futures(double t1, double t2, double implied);
    double maturity() const override;
    const char* type() const override;
    double solveDiscount(const std::function<double(double)>& discount) const override;
};

} // namespace pricing
