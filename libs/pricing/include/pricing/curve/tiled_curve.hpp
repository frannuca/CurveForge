#pragma once
#include <vector>
#include <functional>
#include <stdexcept>

namespace pricing {

/**
 * @brief TiledCurve represents a yield curve parameterized by instantaneous forward rates
 * 
 * This class stores instantaneous forward rates at pillar points and provides
 * discount factors by integrating these forwards. The forward rates are linearly
 * interpolated between pillars, and discount factors are computed using:
 * D(t) = exp(-∫₀ᵗ f(s) ds)
 * where the integral is approximated using the trapezoidal rule.
 */
class TiledCurve {
public:
    /**
     * @brief Construct a TiledCurve from pillar times and instantaneous forward rates
     * @param pillar_times Vector of pillar times (must be strictly increasing and > 0)
     * @param forwards Instantaneous forward rates at each pillar
     */
    TiledCurve(const std::vector<double>& pillar_times, const std::vector<double>& forwards);
    
    /**
     * @brief Get discount factor at time t
     * @param t Time in years (must be >= 0)
     * @return Discount factor D(t)
     */
    double discount(double t) const;
    
    /**
     * @brief Get instantaneous forward rate at time t
     * @param t Time in years (must be >= 0)
     * @return Instantaneous forward rate f(t)
     */
    double instantaneous_forward(double t) const;
    
    /**
     * @brief Get forward rate for period [t, t+dT]
     * @param t Start time in years
     * @param dT Period length in years
     * @return Forward rate F(t, t+dT) = D(t+dT) / D(t)
     */
    double get_forward(double t, double dT) const;
    
    /**
     * @brief Get pillar times
     * @return Vector of pillar times
     */
    const std::vector<double>& pillarTimes() const { return pillar_times_; }
    
    /**
     * @brief Get forward rates at pillars
     * @return Vector of instantaneous forward rates
     */
    const std::vector<double>& forwardRates() const { return forwards_; }

private:
    std::vector<double> pillar_times_;  // Pillar times (strictly increasing)
    std::vector<double> forwards_;       // Instantaneous forward rates at pillars
    
    // Helper: interpolate forward rate at time t
    double interpolate_forward(double t) const;
    
    // Helper: compute integral of forward rates from 0 to t
    double integrate_forward(double t) const;
};

} // namespace pricing
