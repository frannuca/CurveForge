#pragma once

#include "instruments/Instrument.h"
#include <vector>

#include "time/calendars.hpp"
#include "time/date_modifier.hpp"

namespace curve::time {
    enum class FinancialCalendar;
}

namespace curve::instruments {
    class Leg;

    class Swap : public Instrument {
    public:
        Swap() = delete;

        Swap(const Leg &leg1, const Leg &leg2) : leg1_(leg1), leg2_(leg2) {
        };

        ~Swap() override = default;

        std::string name() const override;

        const std::vector<time::Date> &get_leg1_payment_dates() const;

        const std::vector<time::Date> &get_leg2_payment_dates() const;

        double fixedRate() const;

    private:
        const Leg &leg1_;
        const Leg &leg2_;
    };
} // namespace instruments
