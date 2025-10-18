#include "instruments/european_option.h"

namespace curve::instruments {
    EuropeanOption::EuropeanOption(double notional, time::Date start_date, time::Date maturity_date,
                                   const InstrumentBase &underlying, double strike, double maturity)
        : InstrumentBase(maturity_date, start_date), notional_(notional), underlying_(underlying), strike_(strike),
          maturity_(maturity) {
    }

    double EuropeanOption::notional() const { return notional_; }

    double EuropeanOption::pv() const {
        return 0.0; /* placeholder */
    }

    std::string EuropeanOption::name() const { return "EuropeanOption(" + underlying_.name() + ")"; }
} // namespace instruments

