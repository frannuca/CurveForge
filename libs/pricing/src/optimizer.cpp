#include "pricing/curve/optimizer.hpp"
#include <nlopt.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace pricing {

// Wrapper structure to pass data to NLopt objective function
struct OptimizationData {
    const CurveOptimizer* optimizer;
    const std::vector<CalibrationInstrument>* instruments;
    const std::vector<double>* pillar_times;
};

// Static objective function for NLopt (C API)
static double objectiveFunction(unsigned n, const double* x, double* grad, void* data) {
    auto* opt_data = static_cast<OptimizationData*>(data);
    
    // Convert C array to vector
    std::vector<double> forwards(x, x + n);
    
    // Build curve from forward rates
    YieldCurve curve;
    const auto& pillars = *opt_data->pillar_times;
    
    // Convert instantaneous forwards to discount factors
    // D(t) = exp(-integral_0^t f(s) ds)
    // We approximate the integral using trapezoidal rule on the pillar points
    curve.add(pillars[0], std::exp(-forwards[0] * pillars[0]));
    
    for (size_t i = 1; i < pillars.size(); ++i) {
        double dt = pillars[i] - pillars[i-1];
        double avg_forward = 0.5 * (forwards[i-1] + forwards[i]);
        double prev_df = curve.discount(pillars[i-1]);
        double df = prev_df * std::exp(-avg_forward * dt);
        curve.add(pillars[i], df);
    }
    
    // Compute sum of squared errors
    double sum_sq_error = 0.0;
    for (const auto& cal_inst : *opt_data->instruments) {
        // Compute model price using the curve
        try {
            // For curve instruments, the "price" is implicit in the discount factor solving
            // We need to use the curve to compute discount factors and then compute market-implied price
            auto discount_fn = [&curve](double t) { return curve.discount(t); };
            double solved_df = cal_inst.instrument->solveDiscount(discount_fn);
            
            // For calibration, the residual is the difference between market-implied discount
            // and curve-implied discount at maturity
            double maturity = cal_inst.instrument->maturity();
            double curve_df = curve.discount(maturity);
            
            // The error is weighted difference
            double error = (curve_df - solved_df) * cal_inst.weight;
            sum_sq_error += error * error;
        } catch (const std::exception& e) {
            // If pricing fails, add large penalty
            sum_sq_error += 1e6;
        }
    }
    
    return sum_sq_error;
}

CurveOptimizer::CurveOptimizer(const Config& config) 
    : config_(config) {
}

CurveOptimizer::~CurveOptimizer() = default;

CurveOptimizer& CurveOptimizer::add(const CurveInstrumentPtr& instrument, double market_price, double weight) {
    if (!instrument) {
        throw std::invalid_argument("Cannot add null instrument");
    }
    if (weight <= 0.0) {
        throw std::invalid_argument("Weight must be positive");
    }
    instruments_.emplace_back(instrument, market_price, weight);
    return *this;
}

void CurveOptimizer::extractPillarTimes() {
    pillar_times_.clear();
    
    // Extract unique maturity times from instruments
    for (const auto& cal_inst : instruments_) {
        double mat = cal_inst.instrument->maturity();
        pillar_times_.push_back(mat);
    }
    
    // Sort and remove duplicates
    std::sort(pillar_times_.begin(), pillar_times_.end());
    auto last = std::unique(pillar_times_.begin(), pillar_times_.end(),
                           [](double a, double b) { return std::abs(a - b) < 1e-10; });
    pillar_times_.erase(last, pillar_times_.end());
    
    if (pillar_times_.empty()) {
        throw std::runtime_error("No pillar times extracted from instruments");
    }
}

YieldCurve CurveOptimizer::buildCurveFromForwards(const std::vector<double>& forwards) const {
    YieldCurve curve;
    
    if (forwards.size() != pillar_times_.size()) {
        throw std::invalid_argument("Forward vector size must match pillar times");
    }
    
    // Convert instantaneous forwards to discount factors
    curve.add(pillar_times_[0], std::exp(-forwards[0] * pillar_times_[0]));
    
    for (size_t i = 1; i < pillar_times_.size(); ++i) {
        double dt = pillar_times_[i] - pillar_times_[i-1];
        double avg_forward = 0.5 * (forwards[i-1] + forwards[i]);
        double prev_df = curve.discount(pillar_times_[i-1]);
        double df = prev_df * std::exp(-avg_forward * dt);
        curve.add(pillar_times_[i], df);
    }
    
    return curve;
}

double CurveOptimizer::computePrice(const CurveInstrument& inst, const YieldCurve& curve) const {
    auto discount_fn = [&curve](double t) { return curve.discount(t); };
    return inst.solveDiscount(discount_fn);
}

CalibrationResult CurveOptimizer::calibrate() {
    CalibrationResult result;
    result.success = false;
    
    // Extract pillar times from instruments
    extractPillarTimes();
    
    if (instruments_.empty()) {
        result.message = "No instruments to calibrate";
        return result;
    }
    
    // Setup optimization
    size_t n_params = pillar_times_.size();
    nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, n_params);  // Derivative-free local optimizer
    
    if (!opt) {
        result.message = "Failed to create NLopt optimizer";
        return result;
    }
    
    // Set bounds for forward rates (0.1% to 20%)
    std::vector<double> lower_bounds(n_params, 0.001);
    std::vector<double> upper_bounds(n_params, 0.20);
    nlopt_set_lower_bounds(opt, lower_bounds.data());
    nlopt_set_upper_bounds(opt, upper_bounds.data());
    
    // Set stopping criteria
    nlopt_set_ftol_rel(opt, config_.relative_tolerance);
    nlopt_set_ftol_abs(opt, config_.absolute_tolerance);
    nlopt_set_maxeval(opt, config_.max_iterations);
    
    // Set objective function
    OptimizationData opt_data;
    opt_data.optimizer = this;
    opt_data.instruments = &instruments_;
    opt_data.pillar_times = &pillar_times_;
    
    nlopt_set_min_objective(opt, objectiveFunction, &opt_data);
    
    // Initial guess: use configured initial forward rate
    std::vector<double> forwards(n_params, config_.initial_forward_rate);
    
    // Run optimization
    double final_obj_value;
    nlopt_result nlopt_result = nlopt_optimize(opt, forwards.data(), &final_obj_value);
    
    result.objective_value = final_obj_value;
    
    if (nlopt_result >= 0) {
        result.success = true;
        result.message = "Optimization converged successfully";
    } else if (nlopt_result == NLOPT_MAXEVAL_REACHED) {
        result.success = false;
        result.message = "Maximum iterations reached";
    } else {
        result.success = false;
        std::ostringstream oss;
        oss << "Optimization failed with code " << nlopt_result;
        result.message = oss.str();
    }
    
    // Build final curve
    result.curve = buildCurveFromForwards(forwards);
    
    // Compute residuals
    result.residuals.reserve(instruments_.size());
    for (const auto& cal_inst : instruments_) {
        auto discount_fn = [&result](double t) { return result.curve.discount(t); };
        double solved_df = cal_inst.instrument->solveDiscount(discount_fn);
        double maturity = cal_inst.instrument->maturity();
        double curve_df = result.curve.discount(maturity);
        double residual = curve_df - solved_df;
        result.residuals.push_back(residual);
    }
    
    // Clean up
    nlopt_destroy(opt);
    
    return result;
}

} // namespace pricing
