#pragma once

#include "instruments/InstrumentBase.h"

namespace curve::instruments {
    class Bond : public InstrumentBase {
    public:
        Bond(double notional, double couponRate, time::Date start_date, time::Date maturity);

        ~Bond() override = default;

        double notional() const override;

        double pv() const override;

        std::string name() const override;

    private:
        double notional_;
        double couponRate_;
        double maturity_; // years
    };
} // namespace instruments
