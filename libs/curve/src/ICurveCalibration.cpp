//
// Created by Francisco Nunez on 08.11.2025.
//

#include <stdexcept>
#include "curve/ICurveCalibration.h"

namespace curve {
    const ICurve &ICurveCalibration::get_curve() const { return *this; }

    void ICurveCalibration::set_last_pillar(const time::Date &t, double value) {
        if (pillars_.empty()) {
            throw std::runtime_error("No pillars to set last pillar.");
        }
        pillars_.pop_back();
        pillars_.emplace_back(t, value);
    }

    void ICurveCalibration::set_last_pillar(double value) {
        if (pillars_.empty()) {
            throw std::runtime_error("No pillars to set last pillar.");
        }
        const auto new_pillar = pillars_.back().create_new(value);
        // remove the old last element and append the new one to avoid deleted assignment
        pillars_.pop_back();
        pillars_.push_back(new_pillar);
    }
}

