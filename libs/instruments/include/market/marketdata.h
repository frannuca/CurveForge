//
// Created by Francisco Nunez on 21.10.2025.
//

#ifndef CURVEFORGE_MARKETDATA_H
#define CURVEFORGE_MARKETDATA_H
#include <unordered_map>
#include <memory>
#include "curves/ICurve.h"
#include "time/instant.h"
#include "volatility/IVolatility.h"

namespace curve::market {
   class IVolatility;
}

struct MarketData {
   curve::time::Instant snap_time;
   std::unordered_map<std::string::string, std::shared_ptr<curve::ICurve> > curves_ois;
   std::unordered_map<std::string::string, std::shared_ptr<curve::ICurve> > curves_funding;
   std::unordered_map<std::string, double> underlying_spots;
   std::unordered_map<std::string, std::shared_ptr<curve::market::IVolatility> > volatilities;
};
#endif //CURVEFORGE_MARKETDATA_H
