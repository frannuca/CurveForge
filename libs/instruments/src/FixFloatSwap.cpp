//
// Created by Francisco Nunez on 22.11.2025.
//

#include "../include/instruments/FixFloatSwap.h"

curve::instruments::FixFloatSwap::FixFloatSwap(const Leg &fixed_leg, const Leg &floating_leg) : Swap(
    fixed_leg, floating_leg) {
    if (fixed_leg.leg_type() != Leg::FIXED) {
        throw std::invalid_argument("Fixed leg must be fixed.");
    }
    if (floating_leg.leg_type() != Leg::FLOATING) {
        throw std::invalid_argument("Floating leg must be floating.");
    }
}

double curve::instruments::FixFloatSwap::par_rate(const ICurve &discount_curve, const ICurve &forward_curve) const {
    const auto &leg1_schedule = leg1_.cashflows_schedule();
    const auto &leg2_schedule = leg2_.cashflows_schedule();
    const auto &notional1 = leg1_.notional();
    const auto &notional2 = leg2_.notional();

    double pv_leg1 = 0.0;

    for (const auto &period: leg1_schedule.accruals) {
        pv_leg1 += period.accrual * discount_curve.D(period.end_date);
    }


    double pv_leg2 = 0.0;
    bool is_leg2_fix = leg1_.leg_type() == Leg::FIXED;
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
