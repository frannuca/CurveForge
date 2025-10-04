#pragma once
#include "Instrument.hpp"
#include <string>

namespace pricing {
    class CrossCurrencySwap : public Instrument {
    public:
        CrossCurrencySwap(double maturity, double notional, const std::string &ccy1, const std::string &ccy2)
            : maturity_(maturity), notional_(notional), ccy1_(ccy1), ccy2_(ccy2) {
        }

        double maturity() const override { return maturity_; }
        double notional() const override { return notional_; }
        InstrumentType type() const override { return InstrumentType::xSwap; }
        std::string currency1() const { return ccy1_; }
        std::string currency2() const { return ccy2_; }

    private:
        double maturity_;
        double notional_;
        std::string ccy1_;
        std::string ccy2_;
    };
} // namespace pricing

