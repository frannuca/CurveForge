#pragma once
#include "Instrument.hpp"

namespace pricing {
    class Swap : public Instrument {
    public:
        Swap(double maturity, double notional, int legs = 2) : maturity_(maturity), notional_(notional), legs_(legs) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::Swap; }
        int legs() const { return legs_; }

    private:
        double maturity_;
        double notional_;
        int legs_;
    };
} // namespace pricing

