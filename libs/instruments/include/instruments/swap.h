#pragma once

#include "instruments/Instrument.h"
#include <vector>

#include "Leg.h"
#include "time/calendars.hpp"
#include "time/date_modifier.hpp"

namespace curve {
    struct DiscountCurve;
}

namespace curve::time {
    enum class FinancialCalendar;
}

namespace curve::instruments {
    class Swap : public Instrument {
    public:
        Swap() = delete;

        Swap(const Leg &leg1, const Leg &leg2) : leg1_(leg1), leg2_(leg2),
                                                 Instrument(leg1.currency() + '/' + leg2.currency()) {
        };

        ~Swap() override = default;

        std::string name() const override;

        const time::Schedule &get_leg1_payment_dates() const;

        const time::Schedule &get_leg2_payment_dates() const;

        virtual double par_rate(const DiscountCurve &discount_curve, const DiscountCurve &forward_curve) const = 0;

    private:
        const Leg &leg1_;
        const Leg &leg2_;
    };
} // namespace instruments
