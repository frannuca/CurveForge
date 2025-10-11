#pragma once
#include "pricing/curve/instrument.hpp"
#include "pricing/curve/yield_curve.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace pricing {

/**
 * @brief Represents a calibration instrument with market price
 * 
 * This structure holds both the instrument definition and its observed market price.
 */
struct CalibrationInstrument {
    CurveInstrumentPtr instrument;  // The curve instrument
    double market_price;            // Observed market price
    double weight;                  // Weight in optimization (default 1.0)
    
    CalibrationInstrument(CurveInstrumentPtr inst, double mkt_price, double w = 1.0)
        : instrument(std::move(inst)), market_price(mkt_price), weight(w) {}
};

/**
 * @brief Result of curve calibration
 */
struct CalibrationResult {
    YieldCurve curve;               // Calibrated yield curve
    double objective_value;         // Final objective function value (weighted sum of squared errors)
    std::vector<double> residuals;  // Individual residuals for each instrument
    int iterations;                 // Number of optimization iterations
    bool success;                   // Whether optimization converged successfully
    std::string message;            // Status message
};

/**
 * @brief Optimizer for calibrating yield curves to market instrument prices
 * 
 * This class uses NLopt to minimize the least square error between observed
 * market prices and computed prices. The curve is parameterized using instantaneous
 * forward rates at specified pillar points.
 */
class CurveOptimizer {
public:
    /**
     * @brief Configuration for the optimizer
     */
    struct Config {
        double relative_tolerance;      // Relative tolerance for convergence (default 1e-6)
        double absolute_tolerance;      // Absolute tolerance for convergence (default 1e-8)
        int max_iterations;             // Maximum number of iterations (default 1000)
        double initial_forward_rate;    // Initial guess for forward rates (default 0.03)
        
        Config() 
            : relative_tolerance(1e-6)
            , absolute_tolerance(1e-8)
            , max_iterations(1000)
            , initial_forward_rate(0.03) {}
    };
    
    /**
     * @brief Construct optimizer with configuration
     * @param config Optimizer configuration
     */
    explicit CurveOptimizer(const Config& config = Config());
    
    /**
     * @brief Destructor
     */
    ~CurveOptimizer();
    
    /**
     * @brief Add a calibration instrument
     * @param instrument The curve instrument
     * @param market_price Observed market price
     * @param weight Weight in optimization (default 1.0)
     */
    CurveOptimizer& add(const CurveInstrumentPtr& instrument, double market_price, double weight = 1.0);
    
    /**
     * @brief Calibrate the yield curve to minimize pricing errors
     * @return CalibrationResult containing the calibrated curve and diagnostics
     */
    CalibrationResult calibrate();
    
    /**
     * @brief Get the pillar times used for optimization
     * @return Vector of pillar times
     */
    const std::vector<double>& pillarTimes() const { return pillar_times_; }
    
private:
    Config config_;
    std::vector<CalibrationInstrument> instruments_;
    std::vector<double> pillar_times_;
    
    // Helper functions
    void extractPillarTimes();
    YieldCurve buildCurveFromForwards(const std::vector<double>& forwards) const;
    double computePrice(const CurveInstrument& inst, const YieldCurve& curve) const;
};

} // namespace pricing
