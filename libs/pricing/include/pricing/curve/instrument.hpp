#pragma once
#include <functional>
#include <memory>

namespace pricing {

class CurveInstrument {
public:
    virtual ~CurveInstrument() = default;
    virtual double maturity() const = 0;                 // final maturity in years
    virtual const char* type() const = 0;                // instrument tag
    virtual double solveDiscount(const std::function<double(double)>& discount) const = 0; // compute DF at maturity
};

using CurveInstrumentPtr = std::shared_ptr<CurveInstrument>;

} // namespace pricing

