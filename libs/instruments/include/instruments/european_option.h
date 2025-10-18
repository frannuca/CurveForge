#pragma once

#include "instruments/InstrumentBase.h"
#include <string>

namespace curve::instruments {
    class EuropeanOption : public InstrumentBase {
    public:
        EuropeanOption(double notional, time::Date start_date, time::Date maturity_date,
                       const InstrumentBase &underlying, double strike, double maturity);

        ~EuropeanOption() override = default;

        double notional() const override;

        double pv() const override;

        std::string name() const override;

    private:
        double notional_;
        const InstrumentBase &underlying_;
        double strike_;
        double maturity_;
    };
} // namespace instruments
