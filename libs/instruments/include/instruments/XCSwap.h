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

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve,
                                              const ICurve &forward_curve) const override {
            throw std::runtime_error("Not implemented");
        };

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve_leg1, const ICurve &forward_curve_leg1,
                                              const ICurve &discount_curve_leg2,
                                              const ICurve &forward_curve_leg2) const override;

    private:
        double fxSpot_;
    };
}


#endif //CURVEFORGE_XCSWAP_H
