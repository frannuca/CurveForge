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
        ICurve(const time::Date &cob_date, std::vector<Pillar> &&pillars,
               std::shared_ptr<time::DayCountConventionBase> convention);

        ICurve(const time::Date &cob_date, const std::vector<Pillar> &pillars,
               std::shared_ptr<time::DayCountConventionBase> convention);

        ICurve() = delete;

        virtual ~ICurve() = default;

        ICurve(const ICurve &other) = default;

        [[nodiscard]] double D(const time::Date &d) const;

        [[nodiscard]] double F(const time::Date &t1, const time::Date &t2) const;

        [[nodiscard]] virtual std::string name() const =0;

        friend class ICurveCalibration;

    protected:
        std::vector<Pillar> pillars_;
        const time::Date cob_date;
        std::shared_ptr<time::DayCountConventionBase> dc;
    };
} // curve
#endif //CURVEFORGE_ICURVE_H
