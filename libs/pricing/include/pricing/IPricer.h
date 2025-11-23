//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_IPRICER_H
#define CURVEFORGE_IPRICER_H
#include "greeks.h"
#include <memory>
#include "market/marketdata.h"
#include "instruments/Instrument.h"

namespace curve::pricing {
    class IPricer {
    public:
        virtual ~IPricer() = default;

        [[nodiscard]] virtual double pv(const instruments::Instrument &instrument,
                                        std::shared_ptr<market::MarketData> md) const = 0;

        [[nodiscard]] virtual double price(const instruments::Instrument &instrument,
                                           std::shared_ptr<market::MarketData> md) const = 0;

        [[nodiscard]] virtual Greeks compute(const instruments::Instrument &instrument,
                                             std::shared_ptr<market::MarketData> md) const = 0;

        virtual bool CanPriceInstrument(const instruments::Instrument &p) =0;
    };
}


#endif //CURVEFORGE_IPRICER_H
