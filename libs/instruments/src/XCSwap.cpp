//
// Created by Francisco Nunez on 22.11.2025.
//

#include "../include/instruments/XCSwap.h"

curve::instruments::XCSwap::XCSwap(const Leg &base_ccy_leg, const Leg &foreign_ccy_leg) : fxSpot_(
        base_ccy_leg.notional() / foreign_ccy_leg.notional()), Swap(
        base_ccy_leg, foreign_ccy_leg) {
    if (base_ccy_leg.leg_type() != Leg::FLOATING) {
        throw std::invalid_argument("base_ccy_leg leg must be floating.");
    }
    if (foreign_ccy_leg.leg_type() != Leg::FLOATING) {
        throw std::invalid_argument("foreign_ccy_leg leg must be floating.");
    }
}

double curve::instruments::XCSwap::par_rate(const ICurve &discount_curve_leg1, const ICurve &forward_curve_leg1,
                                            const ICurve &discount_curve_leg2,
                                            const ICurve &forward_curve_leg2) const {
    const auto &leg1_schedule = leg1_.cashflows_schedule();
    const auto &leg2_schedule = leg2_.cashflows_schedule();
    const auto &notional1 = leg1_.notional();
    const auto &notional2 = leg2_.notional();

    //Doing base currency leg
    double pv_dom = notional1 * (1 - discount_curve_leg1.D(leg1_schedule.accruals.back().end_date));

    for (const auto &period: leg1_schedule.accruals) {
        const auto D = discount_curve_leg1.D(period.end_date);
        const auto F = forward_curve_leg1.F(period.start_date, period.end_date);
        pv_dom += F * period.accrual * D;
    }


    double pv_foreign = notional2 * (1 - discount_curve_leg2.D(leg2_schedule.accruals.back().end_date));
    bool is_leg2_fix = leg1_.leg_type() == Leg::FIXED;
    for (const auto &period: leg2_schedule.accruals) {
        const auto D = discount_curve_leg2.D(period.end_date);
        const auto F = forward_curve_leg2.F(period.start_date, period.end_date);
        pv_foreign += F * period.accrual * D;
    }
    pv_foreign *= fxSpot_;

    if (pv_dom == 0.0) {
        throw std::runtime_error("PV of leg1 is zero, cannot compute par rate.");
    }


    double annuity = 0.0;
    for (const auto &period: leg2_schedule.accruals) {
        annuity += period.accrual * discount_curve_leg2.D(period.end_date);
    }
    annuity *= (notional2 * fxSpot_);
    double b = (pv_dom - pv_foreign) / annuity;
    return b;
}
