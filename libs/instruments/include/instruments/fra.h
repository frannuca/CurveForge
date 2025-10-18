#pragma once

#include "instruments/InstrumentBase.h"
#include "time/date.hpp"

namespace curve::instruments {
    class FRA : public InstrumentBase {
    public:
        FRA(double notional, time::Date maturity, time::Date start_date, double fixedRate);

        ~FRA() override = default;

        double notional() const override;

        double pv() const override;

        std::string name() const override;

    private:
        double notional_;
        double fixedRate_;
    };
} // namespace instruments
