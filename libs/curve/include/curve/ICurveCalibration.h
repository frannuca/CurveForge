//
// Created by Francisco Nunez on 08.11.2025.
//

#ifndef CURVEFORGE_ICURVECALIBRATION_H
#define CURVEFORGE_ICURVECALIBRATION_H
#include <Eigen/src/Core/util/StaticAssert.h>

#include "ICurve.h"

namespace curve {
    class ICurveCalibration {
    public:
        virtual ~ICurveCalibration() = default;

        ICurveCalibration(ICurve &c);

        virtual void set_last_pillar(const time::Instant &t, double value);

        virtual void set_last_pillar(double value);

        ICurve &get_curve();

    private:
        ICurve &curve_;
    };
}
#endif //CURVEFORGE_ICURVECALIBRATION_H
