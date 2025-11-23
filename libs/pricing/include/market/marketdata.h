//
// Created by Francisco Nunez on 21.10.2025.
//

#ifndef CURVEFORGE_MARKETDATA_H
#define CURVEFORGE_MARKETDATA_H
#include <map>
#include <memory>
#include "curve/ICurve.h"
#include "volatility/IVolatility.h"


namespace curve::market {
    struct MarketData {
        curve::time::Instant snap_time;
        std::map<std::string, std::shared_ptr<ICurve> > curves_ois;
        std::map<std::string, std::shared_ptr<ICurve> > curves_funding;
        std::map<std::string, double> underlying_spots;
        std::map<std::string, std::shared_ptr<curve::market::IVolatility> > volatilities;
    };
}

#endif //CURVEFORGE_MARKETDATA_H
