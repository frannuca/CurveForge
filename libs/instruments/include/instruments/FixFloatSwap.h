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

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve,
                                              const ICurve &forward_curve) const override;

        [[nodiscard]] virtual double par_rate(const ICurve &discount_curve_leg1, const ICurve &forward_curve_leg1,
                                              const ICurve &discount_curve_leg2,
                                              const ICurve &forward_curve_leg2) const override {
            throw std::runtime_error("Not implemented");
        };
    };
};


#endif //CURVEFORGE_FIXFLOATSWAP_H
