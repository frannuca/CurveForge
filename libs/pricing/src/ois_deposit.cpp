#include "pricing/curve/ois_deposit.hpp"
#include <stdexcept>

namespace pricing {

OISDeposit::OISDeposit(double T, double rate) : T_(T), rate_(rate) {
    if (T_ <= 0.0) throw std::invalid_argument("OISDeposit maturity <=0");
}

double OISDeposit::maturity() const { return T_; }
const char* OISDeposit::type() const { return "OIS"; }
double OISDeposit::solveDiscount(const std::function<double(double)>&) const { return 1.0 / (1.0 + rate_ * T_); }

} // namespace pricing

