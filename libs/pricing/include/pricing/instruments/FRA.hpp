#pragma once
#include "Instrument.hpp"

namespace pricing {
    class FRA : public Instrument {
    public:
        FRA(double start, double end, double notional) : start_(start), end_(end), notional_(notional) {
        }

        double maturity() const override { return end_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::Fra; }
        double start() const { return start_; }
        double end() const { return end_; }

    private:
        double start_;
        double end_;
        double notional_;
    };
} // namespace pricing

