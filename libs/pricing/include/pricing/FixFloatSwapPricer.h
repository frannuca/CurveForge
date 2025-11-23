//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_FIXFLOATSWAPPRICER_H
#define CURVEFORGE_FIXFLOATSWAPPRICER_H

#include "IPricer.h"
#include "instruments/Instrument.h"

namespace curve::pricing {
    class FixFloatSwapPricer : public IPricer {
    public:
        [[nodiscard]] virtual double pv(const instruments::Instrument &instrument,
                                        std::shared_ptr<::curve::market::MarketData> md) const override;

        [[nodiscard]] virtual double price(const instruments::Instrument &instrument,
                                           std::shared_ptr<market::MarketData> md) const override;

        [[nodiscard]] virtual curve::pricing::Greeks
        compute(const instruments::Instrument &instrument, std::shared_ptr<market::MarketData> md) const override;
    };
}


#endif //CURVEFORGE_FIXFLOATSWAPPRICER_H
