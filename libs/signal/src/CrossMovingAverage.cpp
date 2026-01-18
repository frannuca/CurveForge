//
// Created by Francisco Nunez on 18.01.2026.
//
#include "signal/CrossMovingAverage.h"

using namespace forge::signal;

CrossMovingAverage::CrossMovingAverage(std::size_t short_period, std::size_t long_period)
    : short_(ExponentialMovingAverage::from_period(short_period)),
      long_(ExponentialMovingAverage::from_period(long_period)) {
}


double CrossMovingAverage::update(double sample) {
    double s = short_.update(sample);
    double l = long_.update(sample);
    double diff = s - l;

    // store last relation and last diff for external inspection
    int relation = 0;
    if (diff < 0.0) relation = -1;
    else if (diff > 0.0) relation = 1;
    last_relation_ = relation;
    last_diff_ = diff;
    return diff;
}
