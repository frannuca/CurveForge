//
// Created by Francisco Nunez on 19.10.2025.
//

#ifndef CURVEFORGE_BASKET_H
#define CURVEFORGE_BASKET_H
#include "Instrument.h"
#include <array>
#include <utility>
#include <functional>

namespace curve {
    namespace instruments {
        struct Component {
            double weight;
            const Instrument &instrument;
        };

        template<std::size_t N>
        struct Underlying {
            std::array<Component, N> instruments;
        };

        template<>
        struct Underlying<1> {
            Component x;
        };
    } // instruments
} // curve

#endif //CURVEFORGE_BASKET_H
