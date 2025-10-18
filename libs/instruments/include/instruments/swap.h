#pragma once

#include "instruments/InstrumentBase.h"
#include <vector>

namespace curve::instruments {
    class Swap : public InstrumentBase {
    public:
        Swap(double notional, time::Date start_date, time::Date mature, const std::vector<time::Date> &fixedSchedule,
             const std::vector<time::Date> &floatSchedule,
             double fixedRate);

        ~Swap() override = default;

        double notional() const override;

        double pv() const override;

        std::string name() const override;

    private:
        double notional_;
        std::vector<time::Date> fixedSchedule_;
        std::vector<time::Date> floatSchedule_;
        double fixedRate_;
    };
} // namespace instruments
