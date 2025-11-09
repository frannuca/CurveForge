//
// Created by Francisco Nunez on 21.10.2025.
//

#ifndef CURVEFORGE_ICURVE_H
#define CURVEFORGE_ICURVE_H
#include <chrono>

#include "Pillar.h"
#include "time/instant.h"

namespace curve {
    class ICurve {
    public:
        virtual double D(const time::Instant &t) const = 0;

        virtual double F(const time::Instant &t, const time::Instant &maturity) const = 0;

        virtual std::string name() const =0;

        friend class ICurveCalibration;

    protected:
        std::vector<Pillar> pillars_;
    };

    class ICalibrationCurve : public ICurve {
    public:
        virtual void set_last(double knot, double v) = 0;
    };
} // curve
#endif //CURVEFORGE_ICURVE_H
