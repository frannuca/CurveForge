#include "instruments/bond.h"

namespace curve::instruments {
    Bond::Bond(double notional, double couponRate, time::Date start_date, time::Date maturity)
        : InstrumentBase(maturity, start_date), notional_(notional), couponRate_(couponRate) {
    }

    double Bond::notional() const { return notional_; }

    double Bond::pv() const {
        return 0.0; /* placeholder */
    }

    std::string Bond::name() const { return "Bond"; }
} // namespace instruments

