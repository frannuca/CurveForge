#include "pricing/bond/bond.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace pricing {
namespace bond {

Bond::Bond(double face_value, double coupon_rate, double maturity, int payment_frequency)
    : face_value_(face_value), coupon_rate_(coupon_rate), maturity_(maturity) {
    
    if (face_value <= 0.0) {
        throw std::invalid_argument("Face value must be positive");
    }
    if (maturity <= 0.0) {
        throw std::invalid_argument("Maturity must be positive");
    }
    if (payment_frequency <= 0) {
        throw std::invalid_argument("Payment frequency must be positive");
    }
    if (coupon_rate < 0.0) {
        throw std::invalid_argument("Coupon rate cannot be negative");
    }

    // Generate coupon payment times and amounts
    double period = 1.0 / payment_frequency;
    double coupon_payment = face_value * coupon_rate / payment_frequency;
    
    for (double t = period; t <= maturity + 1e-10; t += period) {
        coupon_times_.push_back(std::min(t, maturity));
        coupon_amounts_.push_back(coupon_payment);
    }
    
    // Add principal repayment at maturity
    if (!coupon_times_.empty() && std::abs(coupon_times_.back() - maturity) < 1e-10) {
        coupon_amounts_.back() += face_value;
    } else {
        coupon_times_.push_back(maturity);
        coupon_amounts_.push_back(face_value);
    }
}

double Bond::priceFromYield(double yield) const {
    if (coupon_times_.empty()) {
        throw std::logic_error("Bond has no cash flows");
    }

    // Use discrete compounding matching the payment frequency
    // Find the payment frequency from the first period
    double period = coupon_times_[0];
    int frequency = std::round(1.0 / period);
    
    double price = 0.0;
    for (size_t i = 0; i < coupon_times_.size(); ++i) {
        double t = coupon_times_[i];
        double cf = coupon_amounts_[i];
        // Discount factor: (1 + y/freq)^(-freq * t)
        double df = std::pow(1.0 + yield / frequency, -frequency * t);
        price += cf * df;
    }
    
    return price;
}

double Bond::yieldFromPrice(double price, double initial_guess) const {
    if (price <= 0.0) {
        throw std::invalid_argument("Price must be positive");
    }

    // Use coupon rate as initial guess if not provided
    double y = (initial_guess < 0.0) ? coupon_rate_ : initial_guess;
    
    // Find the payment frequency from the first period
    double period = coupon_times_[0];
    int frequency = std::round(1.0 / period);
    
    const int max_iterations = 100;
    const double tolerance = 1e-8;
    
    // Newton-Raphson method
    for (int iter = 0; iter < max_iterations; ++iter) {
        double p = 0.0;      // Price
        double dp_dy = 0.0;  // Derivative of price w.r.t. yield
        
        for (size_t i = 0; i < coupon_times_.size(); ++i) {
            double t = coupon_times_[i];
            double cf = coupon_amounts_[i];
            double n = frequency * t;
            double base = 1.0 + y / frequency;
            double df = std::pow(base, -n);
            
            p += cf * df;
            // d/dy [(1 + y/f)^(-f*t)] = -t * (1 + y/f)^(-f*t-1)
            dp_dy -= cf * t * std::pow(base, -n - 1);
        }
        
        double f = p - price;
        
        if (std::abs(f) < tolerance) {
            return y;
        }
        
        if (std::abs(dp_dy) < 1e-14) {
            throw std::runtime_error("Derivative too small in Newton-Raphson");
        }
        
        y = y - f / dp_dy;
        
        // Keep yield reasonable
        if (y < -0.5 || y > 2.0) {
            throw std::runtime_error("Yield iteration diverged");
        }
    }
    
    throw std::runtime_error("Yield calculation did not converge");
}

double Bond::accruedInterest(double t) const {
    if (t < 0.0 || t > maturity_) {
        throw std::invalid_argument("Time must be between 0 and maturity");
    }

    // Find the previous coupon payment
    size_t idx = 0;
    for (; idx < coupon_times_.size(); ++idx) {
        if (coupon_times_[idx] > t) {
            break;
        }
    }
    
    if (idx == 0) {
        // Before first coupon
        double period = coupon_times_[0];
        return coupon_amounts_[0] * (t / period);
    }
    
    double prev_time = (idx > 0) ? coupon_times_[idx - 1] : 0.0;
    double next_time = (idx < coupon_times_.size()) ? coupon_times_[idx] : maturity_;
    double period = next_time - prev_time;
    double time_since = t - prev_time;
    
    // Only consider coupon payment, not principal
    double coupon_pmt = (idx < coupon_amounts_.size()) ? coupon_amounts_[idx] : 0.0;
    if (idx < coupon_times_.size() && std::abs(coupon_times_[idx] - maturity_) < 1e-10) {
        // Last payment includes principal, subtract it
        coupon_pmt -= face_value_;
    }
    
    return coupon_pmt * (time_since / period);
}

double Bond::duration(double yield) const {
    double price = priceFromYield(yield);
    if (price < 1e-10) {
        throw std::runtime_error("Price too small for duration calculation");
    }
    
    // Find the payment frequency from the first period
    double period = coupon_times_[0];
    int frequency = std::round(1.0 / period);
    
    double weighted_time = 0.0;
    for (size_t i = 0; i < coupon_times_.size(); ++i) {
        double t = coupon_times_[i];
        double cf = coupon_amounts_[i];
        double df = std::pow(1.0 + yield / frequency, -frequency * t);
        double pv = cf * df;
        weighted_time += t * pv;
    }
    
    return weighted_time / price;
}

double Bond::modifiedDuration(double yield) const {
    // For discrete compounding: Modified Duration = Macaulay Duration / (1 + y/freq)
    double period = coupon_times_[0];
    int frequency = std::round(1.0 / period);
    double mac_dur = duration(yield);
    return mac_dur / (1.0 + yield / frequency);
}

double Bond::convexity(double yield) const {
    double price = priceFromYield(yield);
    if (price < 1e-10) {
        throw std::runtime_error("Price too small for convexity calculation");
    }
    
    // Find the payment frequency from the first period
    double period = coupon_times_[0];
    int frequency = std::round(1.0 / period);
    
    double weighted_time_sq = 0.0;
    for (size_t i = 0; i < coupon_times_.size(); ++i) {
        double t = coupon_times_[i];
        double cf = coupon_amounts_[i];
        double df = std::pow(1.0 + yield / frequency, -frequency * t);
        double pv = cf * df;
        weighted_time_sq += t * t * pv;
    }
    
    // Convexity for discrete compounding
    return weighted_time_sq / (price * std::pow(1.0 + yield / frequency, 2));
}

double Bond::priceFromCurve(const std::function<double(double)>& discount_fn) const {
    if (coupon_times_.empty()) {
        throw std::logic_error("Bond has no cash flows");
    }
    
    double price = 0.0;
    for (size_t i = 0; i < coupon_times_.size(); ++i) {
        double t = coupon_times_[i];
        double cf = coupon_amounts_[i];
        double df = discount_fn(t);
        price += cf * df;
    }
    
    return price;
}

} // namespace bond
} // namespace pricing
