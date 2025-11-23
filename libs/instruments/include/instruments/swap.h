#pragma once

#include "instruments/Instrument.h"
#include <vector>

#include "Leg.h"
#include "../../../curve/include/curve/ICurve.h"
#include "time/calendars.hpp"
#include "time/date_modifier.hpp"

namespace curve::instruments {
    class Swap : public Instrument {
    protected:
        Swap() = delete;

        Swap(const Leg &leg1, const Leg &leg2) : leg1_(leg1), leg2_(leg2),
                                                 Instrument(leg1.currency() + '/' + leg2.currency()) {
        };

        ~Swap() override = default;

    public:
        [[nodiscard]] std::string name() const override;

        [[nodiscard]] const time::Schedule &get_leg1_payment_dates() const;

        [[nodiscard]] const time::Schedule &get_leg2_payment_dates() const;

        [[nodiscard]] const Leg &leg1() const { return leg1_; }
        [[nodiscard]] const Leg &leg2() const { return leg2_; }

    protected:
        const Leg &leg1_;
        const Leg &leg2_;
    };
} // namespace instruments
