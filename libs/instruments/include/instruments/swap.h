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

        const time::Schedule &get_leg1_payment_dates() const;

        const time::Schedule &get_leg2_payment_dates() const;

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve,
                                              const ICurve &forward_curve) const = 0;

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve_leg1, const ICurve &forward_curve_leg1,
                                              const ICurve &discount_curve_leg2,
                                              const ICurve &forward_curve_leg2) const = 0;

    protected:
        const Leg &leg1_;
        const Leg &leg2_;
    };
} // namespace instruments
