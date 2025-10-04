#pragma once
#include "Instrument.hpp"

namespace pricing {
    class Bond : public Instrument {
    public:
        Bond(double maturity, double notional, double coupon) : maturity_(maturity), notional_(notional),
                                                                coupon_(coupon) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::Bond; }
        double coupon() const { return coupon_; }

    private:
        double maturity_;
        double notional_;
        double coupon_;
    };
} // namespace pricing

