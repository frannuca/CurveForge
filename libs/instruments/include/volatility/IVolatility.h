//
// Created by Francisco Nunez on 21.10.2025.
//

#ifndef CURVEFORGE_IVOLATILITY_H
#define CURVEFORGE_IVOLATILITY_H
#include "time/instant.h"

namespace curve {
    namespace market {
        class IVolatility {
        public:
            enum class VolatilitySurfaceType {
                STICKY_STRIKE,
                STICKY_DELTA,
            };

            IVolatility(IVolatility::VolatilitySurfaceType type) : type_(type) {
            }

            virtual double volatility(double s, const time::Date &maturity) = 0;

        protected:
            VolatilitySurfaceType type_;
        };
    } // market
} // curve

#endif //CURVEFORGE_IVOLATILITY_H
