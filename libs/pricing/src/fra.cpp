#include "pricing/curve/fra.hpp"
#include <stdexcept>

namespace pricing {

FRA::FRA(double t1, double t2, double fwd) : t1_(t1), t2_(t2), fwd_(fwd) {
    if (!(t2_ > t1_ && t1_ >= 0.0)) throw std::invalid_argument("FRA invalid times");
}

double FRA::maturity() const { return t2_; }
const char* FRA::type() const { return "FRA"; }
double FRA::solveDiscount(const std::function<double(double)>& discount) const {
    double df1 = discount(t1_);
    double dt = t2_ - t1_;
    return df1 / (1.0 + fwd_ * dt);
}

} // namespace pricing

