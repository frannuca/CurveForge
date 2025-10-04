#pragma once
#include "Instrument.hpp"

namespace pricing {
    class EuropeanOption : public Instrument {
    public:
        EuropeanOption(double maturity, double notional, double strike, bool isCall)
            : maturity_(maturity), notional_(notional), strike_(strike), isCall_(isCall) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::Option; }
        double strike() const { return strike_; }
        bool isCall() const { return isCall_; }

    private:
        double maturity_;
        double notional_;
        double strike_;
        bool isCall_;
    };
} // namespace pricing

