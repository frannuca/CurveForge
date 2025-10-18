#include "instruments/swap.h"

namespace curve::instruments {
    Swap::Swap(double notional, time::Date start_date, time::Date maturity,
               const std::vector<time::Date> &fixedSchedule, const std::vector<time::Date> &floatSchedule,
               double fixedRate)
        : InstrumentBase(maturity, start_date), notional_(notional), fixedSchedule_(fixedSchedule),
          floatSchedule_(floatSchedule), fixedRate_(fixedRate) {
    }

    double Swap::notional() const { return notional_; }

    double Swap::pv() const {
        return 0.0; /* placeholder */
    }

    std::string Swap::name() const {
        return "Swap";
    }
} // namespace instruments

