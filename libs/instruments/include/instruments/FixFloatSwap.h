//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_FIXFLOATSWAP_H
#define CURVEFORGE_FIXFLOATSWAP_H
#include "swap.h"

namespace curve::instruments {
    class FixFloatSwap : public Swap {
    public:
        FixFloatSwap(const Leg &fixed_leg, const Leg &floating_leg);
    };
};


#endif //CURVEFORGE_FIXFLOATSWAP_H
