#include "instruments/fra.h"

namespace curve::instruments {
    FRA::FRA(double notional, time::Date maturity, time::Date start_date, double fixedRate)
        : InstrumentBase(maturity, start_date), notional_(notional), fixedRate_(fixedRate) {
    }

    double FRA::notional() const { return notional_; }

    double FRA::pv() const {
        return 0.0; /* placeholder */
    }

    std::string FRA::name() const {
        return "FRA(" + time::to_string(start_date) + "," + time::to_string(maturity_date) + ")";
    }
} // namespace instruments

