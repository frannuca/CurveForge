#pragma once
#include "pricing/curve/instrument.hpp"
#include <stdexcept>
#include <vector>
#include <cmath>

namespace pricing {

// Par fixed-for-floating swap (payer fixed) used for bootstrapping a single discount curve.
class IRSwap : public CurveInstrument {
    std::vector<double> payTimes_; // ascending >0
    double fixedRate_;
public:
    IRSwap(std::vector<double> payTimes, double fixedRate);
    double maturity() const override;
    const char* type() const override;
    double solveDiscount(const std::function<double(double)>& discount) const override;
    const std::vector<double>& paymentTimes() const;
};

} // namespace pricing
