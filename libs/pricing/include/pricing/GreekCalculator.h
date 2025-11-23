//
// Created by Francisco Nunez on 23.11.2025.
//

#ifndef CURVEFORGE_GREEKCALCULATOR_H
#define CURVEFORGE_GREEKCALCULATOR_H

#include "instruments/Instrument.h"

namespace curve::pricing {
    class GreekCalculator {
    public:
        static double dv01(const curve::instruments::Instrument &p);

        static double delta(const curve::instruments::Instrument &p);

        static double gamma(const curve::instruments::Instrument &p);
    };
}


#endif //CURVEFORGE_GREEKCALCULATOR_H
