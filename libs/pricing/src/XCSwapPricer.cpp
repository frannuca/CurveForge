//
// Created by Francisco Nunez on 22.11.2025.
//

#include "pricing/XCSwapPricer.h"
#include "instruments/XCSwap.h"

namespace curve::pricing {
    double XCSwapPricer::pv(const instruments::Instrument &instrument,
                            std::shared_ptr<market::MarketData> md) const {
        throw std::runtime_error("Not implemented.");
    }

    Greeks XCSwapPricer::compute(const instruments::Instrument &instrument,
                                 std::shared_ptr<market::MarketData> md) const {
        throw std::runtime_error("Not implemented.");
    }

    double XCSwapPricer::price(const instruments::Instrument &instrument,
                               std::shared_ptr<market::MarketData> md) const {
        const instruments::XCSwap *swap = dynamic_cast<const instruments::XCSwap *>(&instrument);
        if (swap == nullptr) { throw std::runtime_error("Instrument is not an XCSwap."); }

        if (swap->leg1().leg_type() != instruments::Leg::FLOATING || swap->leg2().leg_type() !=
            instruments::Leg::FLOATING) {
            throw std::invalid_argument("XCSwap must have floating legs.");
        };

        const auto &leg1_schedule = swap->get_leg1_payment_dates();
        const auto &leg2_schedule = swap->get_leg2_payment_dates();

        const auto &notional1 = swap->leg1().notional();
        const auto &notional2 = swap->leg2().notional();


        const std::shared_ptr<ICurve> discount_curve_leg1 = md->curves_ois.at(swap->leg1().currency());
        const std::shared_ptr<ICurve> forward_curve_leg1 = md->curves_funding.at(swap->leg1().currency());
        //Doing base currency leg
        double pv_dom = notional1 * (1 - discount_curve_leg1->D(leg1_schedule.accruals.back().end_date));

        for (const auto &period: leg1_schedule.accruals) {
            const auto D = discount_curve_leg1->D(period.end_date);
            const auto F = forward_curve_leg1->F(period.start_date, period.end_date);
            pv_dom += F * period.accrual * D;
        }

        const std::shared_ptr<ICurve> discount_curve_leg2 = md->curves_ois.at(swap->leg2().currency());
        const std::shared_ptr<ICurve> forward_curve_leg2 = md->curves_funding.at(swap->leg2().currency());
        double pv_foreign = notional2 * (1 - discount_curve_leg2->D(leg2_schedule.accruals.back().end_date));

        for (const auto &period: leg2_schedule.accruals) {
            const auto D = discount_curve_leg2->D(period.end_date);
            const auto F = forward_curve_leg2->F(period.start_date, period.end_date);
            pv_foreign += F * period.accrual * D;
        }
        pv_foreign *= swap->fxSpot();

        if (pv_dom == 0.0) {
            throw std::runtime_error("PV of leg1 is zero, cannot compute par rate.");
        }


        double annuity = 0.0;
        for (const auto &period: leg2_schedule.accruals) {
            annuity += period.accrual * discount_curve_leg2->D(period.end_date);
        }
        annuity *= (notional2 * swap->fxSpot());
        double b = (pv_dom - pv_foreign) / annuity;
        return b;
    }
}

