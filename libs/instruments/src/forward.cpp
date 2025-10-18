#include "instruments/forward.h"

namespace curve::instruments {
    Forward::Forward(double notional, time::Date start_date, time::Date maturity_date, const InstrumentBase &underlying)
        : InstrumentBase(maturity_date, start_date), underlying_(underlying) {
    }

    double Forward::notional() const { return notional_; }

    double Forward::pv() const {
        return 0.0; /* placeholder: PV requires curve/market data */
    }

    std::string Forward::name() const { return "Forward(" + underlying_.name() + ")"; }
} // namespace instruments

