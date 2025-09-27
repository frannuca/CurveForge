#include "pricing/curve/futures.hpp"
#include <stdexcept>

namespace pricing {

Futures::Futures(double t1, double t2, double implied) : t1_(t1), t2_(t2), implied_(implied) {
    if (!(t2_ > t1_ && t1_ >= 0.0)) throw std::invalid_argument("Futures invalid times");
}

double Futures::maturity() const { return t2_; }
const char* Futures::type() const { return "FUT"; }
double Futures::solveDiscount(const std::function<double(double)>& discount) const {
    double df1 = discount(t1_);
    double dt = t2_ - t1_;
    return df1 / (1.0 + implied_ * dt);
}

} // namespace pricing

