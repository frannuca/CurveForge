//
// Created by Francisco Nunez on 08.11.2025.
//

#ifndef CURVEFORGE_ICURVECALIBRATION_H
#define CURVEFORGE_ICURVECALIBRATION_H
#include <Eigen/src/Core/util/StaticAssert.h>

#include "ICurve.h"

namespace curve {
    class ICurveCalibration : ICurve {
    public:
        virtual ~ICurveCalibration() = default;

        virtual void set_last_pillar(const time::Date &t, double value);

        virtual void set_last_pillar(double value);

        const ICurve &get_curve() const;
    };
}
#endif //CURVEFORGE_ICURVECALIBRATION_H
