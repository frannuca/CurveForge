#pragma once
#include "Instrument.hpp"

namespace pricing {
    class CDS : public Instrument {
    public:
        CDS(double maturity, double notional, double spread) : maturity_(maturity), notional_(notional),
                                                               spread_(spread) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::Cds; }
        double spread() const { return spread_; }

    private:
        double maturity_;
        double notional_;
        double spread_;
    };
} // namespace pricing

