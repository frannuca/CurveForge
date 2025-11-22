//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_FLATRATECURVE_H
#define CURVEFORGE_FLATRATECURVE_H
#include "ICurve.h"
#include "time/date.hpp"

namespace curve {
    //Aple mock curve for testing that returns a constant value.
    class FlatRateCurve : public ICurve {
    public:
        explicit FlatRateCurve(time::Date cob_date, double constant_rate);

        [[nodiscard]] std::string name() const override;

    private:
        double constant_rate_;
    };
}

#endif //CURVEFORGE_FLATRATECURVE_H
