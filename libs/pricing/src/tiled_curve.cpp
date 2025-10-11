#include "pricing/curve/tiled_curve.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace pricing {

TiledCurve::TiledCurve(const std::vector<double>& pillar_times, const std::vector<double>& forwards)
    : pillar_times_(pillar_times), forwards_(forwards) {
    
    // Validate inputs
    if (pillar_times_.empty() || forwards_.empty()) {
        throw std::invalid_argument("TiledCurve: pillar times and forwards cannot be empty");
    }
    
    if (pillar_times_.size() != forwards_.size()) {
        throw std::invalid_argument("TiledCurve: pillar times and forwards must have same size");
    }
    
    // Check that pillar times are strictly increasing and positive
    for (size_t i = 0; i < pillar_times_.size(); ++i) {
        if (pillar_times_[i] <= 0.0) {
            throw std::invalid_argument("TiledCurve: pillar times must be positive");
        }
        if (i > 0 && pillar_times_[i] <= pillar_times_[i-1]) {
            throw std::invalid_argument("TiledCurve: pillar times must be strictly increasing");
        }
    }
}

double TiledCurve::interpolate_forward(double t) const {
    if (t <= 0.0) {
        return forwards_[0];
    }
    
    // If t is at or before first pillar, use first forward rate
    if (t <= pillar_times_[0]) {
        return forwards_[0];
    }
    
    // If t is at or after last pillar, use last forward rate
    if (t >= pillar_times_.back()) {
        return forwards_.back();
    }
    
    // Linear interpolation between pillars
    for (size_t i = 1; i < pillar_times_.size(); ++i) {
        if (t <= pillar_times_[i]) {
            double t0 = pillar_times_[i-1];
            double t1 = pillar_times_[i];
            double f0 = forwards_[i-1];
            double f1 = forwards_[i];
            
            double weight = (t - t0) / (t1 - t0);
            return f0 + weight * (f1 - f0);
        }
    }
    
    // Should never reach here
    return forwards_.back();
}

double TiledCurve::integrate_forward(double t) const {
    if (t <= 0.0) {
        return 0.0;
    }
    
    double integral = 0.0;
    double prev_t = 0.0;
    double prev_f = forwards_[0];
    
    // Integrate from 0 to t using trapezoidal rule
    for (size_t i = 0; i < pillar_times_.size(); ++i) {
        double curr_t = pillar_times_[i];
        double curr_f = forwards_[i];
        
        if (curr_t >= t) {
            // Interpolate forward at t
            double f_at_t = interpolate_forward(t);
            // Add final segment from prev_t to t
            double dt = t - prev_t;
            integral += 0.5 * (prev_f + f_at_t) * dt;
            break;
        }
        
        // Add full segment from prev_t to curr_t
        double dt = curr_t - prev_t;
        integral += 0.5 * (prev_f + curr_f) * dt;
        
        prev_t = curr_t;
        prev_f = curr_f;
    }
    
    // If t is beyond last pillar, extrapolate with constant forward
    if (t > pillar_times_.back()) {
        double dt = t - pillar_times_.back();
        integral += forwards_.back() * dt;
    }
    
    return integral;
}

double TiledCurve::discount(double t) const {
    if (t < 0.0) {
        throw std::invalid_argument("TiledCurve::discount: time must be non-negative");
    }
    
    if (t == 0.0) {
        return 1.0;
    }
    
    double integral = integrate_forward(t);
    return std::exp(-integral);
}

double TiledCurve::instantaneous_forward(double t) const {
    if (t < 0.0) {
        throw std::invalid_argument("TiledCurve::instantaneous_forward: time must be non-negative");
    }
    
    return interpolate_forward(t);
}

double TiledCurve::get_forward(double t, double dT) const {
    if (t < 0.0 || dT <= 0.0) {
        throw std::invalid_argument("TiledCurve::get_forward: invalid time or period");
    }
    
    double df_t = discount(t);
    double df_t_plus_dT = discount(t + dT);
    
    return df_t_plus_dT / df_t;
}

} // namespace pricing
