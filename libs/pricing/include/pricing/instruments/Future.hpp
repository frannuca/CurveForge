#pragma once
#include "Instrument.hpp"

namespace pricing {
    class Future : public Instrument {
    public:
        Future(double maturity, double notional) : maturity_(maturity), notional_(notional) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::Future; }

    private:
        double maturity_;
        double notional_;
    };
} // namespace pricing

