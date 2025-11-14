//
// Created by Francisco Nunez on 21.10.2025.
//

#ifndef CURVEFORGE_ICURVE_H
#define CURVEFORGE_ICURVE_H
#include <chrono>

#include "Pillar.h"
#include "time/daycount.hpp"
#include "time/instant.h"

namespace curve {
    class ICurve {
    public:
        ICurve(std::vector<Pillar> &&pillars);

        ICurve(const std::vector<Pillar> &pillars);

        ICurve() = delete;

        virtual ~ICurve() = default;

        ICurve(const ICurve &other) = default;

        double D(const std::chrono::days &t) const;

        double F(const time::Date t0, const std::chrono::days &dt1, const std::chrono::days &tenor,
                 const time::DayCountConventionBase &dc) const;

        double F(const time::Date t0, const std::chrono::days &dt1, const std::chrono::months &tenor,
                 const time::DayCountConventionBase &dc) const;


        virtual std::string name() const =0;

        friend class ICurveCalibration;

    protected:
        std::vector<Pillar> pillars_;
    };
} // curve
#endif //CURVEFORGE_ICURVE_H
