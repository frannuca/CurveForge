#include "instruments/swap.h"

#include "instruments/Leg.h"
#include "time/calendar_factory.hpp"
#include "time/scheduler.h"

namespace curve::instruments {
    const time::Schedule &Swap::get_leg1_payment_dates() const { return leg1_.cashflows_schedule(); }

    const time::Schedule &Swap::get_leg2_payment_dates() const { return leg2_.cashflows_schedule(); }


    double Swap::par_rate(const ICurve &discount_curve, const ICurve &forward_curve) const {
        const auto &leg1_schedule = leg1_.cashflows_schedule();
        const auto &leg2_schedule = leg2_.cashflows_schedule();
        const auto &notional1 = leg1_.notional();
        const auto &notional2 = leg2_.notional();

        double pv_leg1 = 0.0;
        for (const auto &period: leg1_schedule.accruals) {
            const auto D = discount_curve.D(period.end_date);
            pv_leg1 += period.accrual * D;
        }

        double pv_leg2 = 0.0;
        for (const auto &period: leg2_schedule.accruals) {
            const auto D = discount_curve.D(period.end_date);
            const auto F = forward_curve.F(period.start_date, period.end_date);
            pv_leg2 += F * period.accrual * D;
        }

        if (pv_leg1 == 0.0) {
            throw std::runtime_error("PV of leg1 is zero, cannot compute par rate.");
        }

        return notional2 * pv_leg2 / (notional1 * pv_leg1);
    }

    std::string Swap::name() const {
        return "swap";
    }
} // namespace instruments

