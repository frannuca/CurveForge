#pragma once
#include "Instrument.hpp"

namespace pricing {
    class BarrierOption : public Instrument {
    public:
        BarrierOption(double maturity, double notional, double strike, double barrier, bool isCall)
            : maturity_(maturity), notional_(notional), strike_(strike), barrier_(barrier), isCall_(isCall) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::BarrierOption; }
        double strike() const { return strike_; }
        double barrier() const { return barrier_; }
        bool isCall() const { return isCall_; }

    private:
        double maturity_;
        double notional_;
        double strike_;
        double barrier_;
        bool isCall_;
    };
} // namespace pricing

