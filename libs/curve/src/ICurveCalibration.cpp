//
// Created by Francisco Nunez on 08.11.2025.
//

#include <stdexcept>
#include "curve/ICurveCalibration.h"

namespace curve {
    ICurveCalibration::ICurveCalibration(ICurve &c) : curve_(c) {
    }

    ICurve &ICurveCalibration::get_curve() { return curve_; }

    void ICurveCalibration::set_last_pillar(const time::Instant &t, double value) {
        if (curve_.pillars_.empty()) {
            throw std::runtime_error("No pillars to set last pillar.");
        }
        curve_.pillars_.pop_back();
        curve_.pillars_.push_back(Pillar(t, value));
    }

    void ICurveCalibration::set_last_pillar(double value) {
        if (curve_.pillars_.empty()) {
            throw std::runtime_error("No pillars to set last pillar.");
        }
        auto new_pillar = curve_.pillars_.back().create_new(value);
        // remove the old last element and append the new one to avoid deleted assignment
        curve_.pillars_.pop_back();
        curve_.pillars_.push_back(std::move(new_pillar));
    }
}

