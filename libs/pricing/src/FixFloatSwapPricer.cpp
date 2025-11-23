//
// Created by Francisco Nunez on 22.11.2025.
//
#include "pricing/FixFloatSwapPricer.h"
#include "instruments/FixFloatSwap.h"

namespace curve::pricing {
    using namespace curve::instruments;

    double FixFloatSwapPricer::pv(const instruments::Instrument &instrument,
                                  std::shared_ptr<market::MarketData> md) const {
        throw std::runtime_error("Not implemented.");
    }

    Greeks FixFloatSwapPricer::compute(const instruments::Instrument &instrument,
                                       std::shared_ptr<market::MarketData> md) const {
        throw std::runtime_error("Not implemented.");
    }

    double FixFloatSwapPricer::price(const Instrument &instrument,
                                     std::shared_ptr<market::MarketData> md) const {
        const FixFloatSwap *swap = dynamic_cast<const FixFloatSwap *>(&instrument);
        if (swap == nullptr) { throw std::runtime_error("Instrument is not a FixFloatSwap."); }

        const auto &leg1_schedule = swap->get_leg1_payment_dates();
        const auto &leg2_schedule = swap->get_leg2_payment_dates();
        const auto &notional1 = swap->leg1().notional();
        const auto &notional2 = swap->leg2().notional();

        std::shared_ptr<ICurve> ois1 = md->curves_ois.at(swap->leg1().currency());
        double pv_leg1 = 0.0;

        for (const auto &period: leg1_schedule.accruals) {
            pv_leg1 += period.accrual * ois1->D(period.end_date);
        }

        std::shared_ptr<ICurve> ois2 = md->curves_ois.at(swap->leg2().currency());
        std::shared_ptr<ICurve> forward_curve = md->curves_funding.at(swap->leg2().currency());
        double pv_leg2 = 0.0;

        for (const auto &period: leg2_schedule.accruals) {
            const auto D = ois2->D(period.end_date);
            const auto F = forward_curve->F(period.start_date, period.end_date);
            pv_leg2 += F * period.accrual * D;
        }

        if (pv_leg1 == 0.0) {
            throw std::runtime_error("PV of leg1 is zero, cannot compute par rate.");
        }

        return notional2 * pv_leg2 / (notional1 * pv_leg1);
    }
}
