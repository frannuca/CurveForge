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
    YieldCurve curve;                  // Calibrated yield curve
    double objective_value;            // Final objective function value (weighted sum of squared errors)
    std::vector<double> residuals;     // Individual residuals for each instrument
    int iterations;                    // Number of optimization iterations
    bool success;                      // Whether optimization converged successfully
    std::string message;               // Status message
    std::vector<double> pillar_times;  // Pillar times used for calibration
    std::vector<double> forward_rates; // Calibrated instantaneous forward rates at pillars
};

/**
 * @brief Optimizer for calibrating yield curves to market instrument prices
 * 
 * This class uses NLopt's SLSQP (Sequential Least Squares Programming) algorithm,
 * which is a gradient-based Sequential Quadratic Programming (SQP) method, to minimize
 * the least square error between observed market prices and computed prices. The curve
 * is parameterized using instantaneous forward rates at specified pillar points.
 * 
 * The optimizer supports curve regularization to promote smooth forward curves by
 * penalizing rapid changes in forward rates (first-order) or their slope (second-order).
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
        double regularization_lambda;   // Regularization strength for curve smoothing (default 0.01)
        int regularization_order;       // Order of regularization: 1 (first derivative) or 2 (second derivative, default)
        
        Config() 
            : relative_tolerance(1e-6)
            , absolute_tolerance(1e-8)
            , max_iterations(1000)
            , initial_forward_rate(0.03)
            , regularization_lambda(0.01)
            , regularization_order(2) {}
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
