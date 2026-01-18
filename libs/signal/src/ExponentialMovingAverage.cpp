//
// Created by Francisco Nunez on 18.01.2026.
//

#include "signal/ExponentialMovingAverage.h"

#include <cassert>
using namespace forge::signal;


ExponentialMovingAverage ExponentialMovingAverage::from_period(std::size_t period) {
    assert(period >= 1);
    double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    return ExponentialMovingAverage(alpha);
}


// Update the EMA with a new sample and return the updated EMA value.
// If the EMA has no prior value, the first sample initializes it.
double ExponentialMovingAverage::update(double sample) {
    if (!value_.has_value()) {
        value_ = sample;
    } else {
        value_ = alpha_ * sample + (1.0 - alpha_) * (*value_);
    }
    return *value_;
}
