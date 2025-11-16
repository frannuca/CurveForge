#pragma once

#include "instruments/Instrument.h"
#include <vector>

#include "Leg.h"
#include "../../../curve/include/curve/ICurve.h"
#include "time/calendars.hpp"
#include "time/date_modifier.hpp"

namespace curve::instruments {
    class Swap : public Instrument {
    public:
        Swap() = delete;

        Swap(const Leg &leg1, const Leg &leg2) : leg1_(leg1), leg2_(leg2),
                                                 Instrument(leg1.currency() + '/' + leg2.currency()) {
        };

        ~Swap() override = default;

        [[nodiscard]] std::string name() const override;

        const time::Schedule &get_leg1_payment_dates() const;

        const time::Schedule &get_leg2_payment_dates() const;

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve,
                                              const ICurve &forward_curve) const;

    private:
        const Leg &leg1_;
        const Leg &leg2_;
    };
} // namespace instruments
