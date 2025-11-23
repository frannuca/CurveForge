//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_XCSWAPPRICER_H
#define CURVEFORGE_XCSWAPPRICER_H
#include "IPricer.h"
#include "instruments/Instrument.h"
#include "market/marketdata.h"

namespace curve::instruments {
    class XCSwap;
}

namespace curve::pricing {
    class XCSwapPricer : public IPricer {
    public:
        [[nodiscard]] virtual double pv(const instruments::Instrument &instrument,
                                        std::shared_ptr<market::MarketData> md) const override;

        [[nodiscard]] virtual double price(const instruments::Instrument &instrument,
                                           std::shared_ptr<market::MarketData> md) const override;

        [[nodiscard]] virtual Greeks compute(const instruments::Instrument &instrument,
                                             std::shared_ptr<market::MarketData> md) const override;

        [[nodiscard]] bool CanPriceInstrument(const instruments::Instrument &p) override;

    protected:
        static bool registered;
    };
}


#endif //CURVEFORGE_XCSWAPPRICER_H
