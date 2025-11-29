//
// Created by Francisco Nunez on 29.11.2025.
//

#ifndef CURVEFORGE_VOLATILITYBASE_H
#define CURVEFORGE_VOLATILITYBASE_H
#include "datacontracts/marketdata.hxx"
#include "datacontracts/vol.hxx"

namespace curve::volatility {
    class VolatilityFactory {
    public:
        virtual ~VolatilityFactory() = default;

        virtual std::unique_ptr<vol::VolSurface> Calibrate(const marketdata::MarketDataSnapshot &md) = 0;
    };
}


#endif //CURVEFORGE_VOLATILITYBASE_H
