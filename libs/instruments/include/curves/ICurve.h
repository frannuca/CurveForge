//
// Created by Francisco Nunez on 21.10.2025.
//

#ifndef CURVEFORGE_ICURVE_H
#define CURVEFORGE_ICURVE_H
#include <chrono>
#include "time/instant.h"

namespace curve::curves {
    class ICurve {
    public:
        virtual double zero(const time::Instant &t) = 0;

        virtual double forward(const time::Instant &t, const time::Instant &maturity) = 0;

        virtual std::string name() =0;
    };
} // curve
#endif //CURVEFORGE_ICURVE_H
