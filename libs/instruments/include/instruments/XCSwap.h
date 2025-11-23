//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_XCSWAP_H
#define CURVEFORGE_XCSWAP_H
#include "swap.h"


namespace curve::instruments {
    class XCSwap : public Swap {
    public:
        XCSwap(const Leg &base_ccy_leg, const Leg &foreign_ccy_leg);

        [[nodiscard]] double fxSpot() const;

    private:
        double fxSpot_;
    };
}


#endif //CURVEFORGE_XCSWAP_H
