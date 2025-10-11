/**
 * @file optimizer_example.cpp
 * @brief Demonstrates usage of the CurveForge optimization library
 * 
 * This example shows how to calibrate yield curves using various market instruments
 * including deposits, FRAs, and interest rate swaps.
 */

#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include "pricing/curve/fra.hpp"
#include "pricing/curve/irswap.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace pricing;

void printSeparator() {
    std::cout << std::string(80, '=') << std::endl;
}

void printCurve(const YieldCurve& curve, const std::vector<double>& times) {
    std::cout << std::setw(10) << "Time (Y)" 
              << std::setw(15) << "Discount" 
              << std::setw(15) << "Zero Rate (%)"
              << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (double t : times) {
        double df = curve.discount(t);
        double zero_rate = -std::log(df) / t * 100.0;
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << t
                  << std::setw(15) << std::setprecision(6) << df
                  << std::setw(15) << std::setprecision(4) << zero_rate
                  << std::endl;
    }
}

/**
 * Example 1: Simple deposit curve
 */
void example1_SimpleDeposits() {
    printSeparator();
    std::cout << "EXAMPLE 1: Simple Deposit Curve" << std::endl;
    printSeparator();
    
    CurveOptimizer optimizer;
    
    // Market data: OIS deposits
    struct MarketData {
        double tenor;
        double rate;
    };
    
    std::vector<MarketData> deposits = {
        {0.25, 0.0250},  // 3M at 2.50%
        {0.50, 0.0280},  // 6M at 2.80%
        {1.00, 0.0300},  // 1Y at 3.00%
        {2.00, 0.0320},  // 2Y at 3.20%
        {3.00, 0.0340},  // 3Y at 3.40%
    };
    
    std::cout << "\nMarket Data (OIS Deposits):" << std::endl;
    std::cout << std::setw(10) << "Tenor (Y)" << std::setw(15) << "Rate (%)" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    
    for (const auto& data : deposits) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << data.tenor
                  << std::setw(15) << std::setprecision(4) << (data.rate * 100.0) << std::endl;
        
        auto deposit = std::make_shared<OISDeposit>(data.tenor, data.rate);
        optimizer.add(deposit, 0.0, 1.0);
    }
    
    // Calibrate
    std::cout << "\nCalibrating curve..." << std::endl;
    auto result = optimizer.calibrate();
    
    std::cout << "\nCalibration Results:" << std::endl;
    std::cout << "  Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "  Message: " << result.message << std::endl;
    std::cout << "  Objective Value: " << std::scientific << result.objective_value << std::endl;
    std::cout << std::fixed;
    
    std::cout << "\nCalibrated Curve:" << std::endl;
    std::vector<double> display_times = {0.25, 0.5, 1.0, 2.0, 3.0};
    printCurve(result.curve, display_times);
    
    std::cout << "\nResiduals (DF errors):" << std::endl;
    for (size_t i = 0; i < result.residuals.size(); ++i) {
        std::cout << "  Instrument " << i << ": " 
                  << std::scientific << result.residuals[i] << std::endl;
    }
    std::cout << std::fixed;
}

/**
 * Example 2: Mixed instrument curve (deposits + FRAs)
 */
void example2_MixedInstruments() {
    printSeparator();
    std::cout << "EXAMPLE 2: Mixed Instrument Curve (Deposits + FRAs)" << std::endl;
    printSeparator();
    
    CurveOptimizer optimizer;
    
    // Short-end: deposits
    std::cout << "\nShort-End Deposits:" << std::endl;
    optimizer.add(std::make_shared<OISDeposit>(0.25, 0.0250), 0.0, 1.0);
    std::cout << "  3M: 2.50%" << std::endl;
    
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.0275), 0.0, 1.0);
    std::cout << "  6M: 2.75%" << std::endl;
    
    // Mid-section: FRAs
    std::cout << "\nForward Rate Agreements:" << std::endl;
    optimizer.add(std::make_shared<FRA>(0.5, 1.0, 0.0295), 0.0, 1.0);
    std::cout << "  6Mx12M: 2.95%" << std::endl;
    
    optimizer.add(std::make_shared<FRA>(1.0, 1.5, 0.0310), 0.0, 1.0);
    std::cout << "  12Mx18M: 3.10%" << std::endl;
    
    optimizer.add(std::make_shared<FRA>(1.5, 2.0, 0.0325), 0.0, 1.0);
    std::cout << "  18Mx24M: 3.25%" << std::endl;
    
    // Calibrate
    std::cout << "\nCalibrating curve..." << std::endl;
    auto result = optimizer.calibrate();
    
    std::cout << "\nCalibration Results:" << std::endl;
    std::cout << "  Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "  Objective Value: " << std::scientific << result.objective_value << std::endl;
    std::cout << std::fixed;
    
    std::cout << "\nCalibrated Curve:" << std::endl;
    std::vector<double> display_times = {0.25, 0.5, 1.0, 1.5, 2.0};
    printCurve(result.curve, display_times);
}

/**
 * Example 3: Full curve with swaps
 */
void example3_FullCurveWithSwaps() {
    printSeparator();
    std::cout << "EXAMPLE 3: Full Curve with Swaps" << std::endl;
    printSeparator();
    
    // Custom configuration for better convergence
    CurveOptimizer::Config config;
    config.max_iterations = 1000;
    config.relative_tolerance = 1e-7;
    config.initial_forward_rate = 0.03;
    
    CurveOptimizer optimizer(config);
    
    // Short-end: deposits
    std::cout << "\nDeposits:" << std::endl;
    std::cout << "  3M: 2.50%, 6M: 2.75%, 1Y: 3.00%" << std::endl;
    optimizer.add(std::make_shared<OISDeposit>(0.25, 0.0250), 0.0, 2.0);  // Higher weight
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.0275), 0.0, 2.0);
    optimizer.add(std::make_shared<OISDeposit>(1.0, 0.0300), 0.0, 2.0);
    
    // Long-end: swaps
    std::cout << "\nInterest Rate Swaps:" << std::endl;
    
    // 2Y swap at 3.20%
    std::vector<double> swap2y = {1.0, 2.0};
    optimizer.add(std::make_shared<IRSwap>(swap2y, 0.0320), 0.0, 1.0);
    std::cout << "  2Y: 3.20%" << std::endl;
    
    // 5Y swap at 3.50%
    std::vector<double> swap5y = {1.0, 2.0, 3.0, 4.0, 5.0};
    optimizer.add(std::make_shared<IRSwap>(swap5y, 0.0350), 0.0, 1.0);
    std::cout << "  5Y: 3.50%" << std::endl;
    
    // 10Y swap at 3.80%
    std::vector<double> swap10y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    optimizer.add(std::make_shared<IRSwap>(swap10y, 0.0380), 0.0, 1.0);
    std::cout << "  10Y: 3.80%" << std::endl;
    
    // Calibrate
    std::cout << "\nCalibrating curve..." << std::endl;
    auto result = optimizer.calibrate();
    
    std::cout << "\nCalibration Results:" << std::endl;
    std::cout << "  Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "  Message: " << result.message << std::endl;
    std::cout << "  Objective Value: " << std::scientific << result.objective_value << std::endl;
    std::cout << std::fixed;
    
    std::cout << "\nCalibrated Curve (selected maturities):" << std::endl;
    std::vector<double> display_times = {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    printCurve(result.curve, display_times);
    
    std::cout << "\nNumber of calibration pillars: " << optimizer.pillarTimes().size() << std::endl;
}

/**
 * Example 4: Weighted calibration
 */
void example4_WeightedCalibration() {
    printSeparator();
    std::cout << "EXAMPLE 4: Weighted Calibration" << std::endl;
    printSeparator();
    
    std::cout << "\nDemonstrating the effect of weights on calibration..." << std::endl;
    std::cout << "We'll calibrate twice: once with equal weights, once emphasizing short-end" << std::endl;
    
    // Scenario 1: Equal weights
    std::cout << "\n--- Scenario 1: Equal Weights ---" << std::endl;
    CurveOptimizer opt1;
    opt1.add(std::make_shared<OISDeposit>(0.5, 0.0280), 0.0, 1.0);
    opt1.add(std::make_shared<OISDeposit>(5.0, 0.0380), 0.0, 1.0);
    
    auto result1 = opt1.calibrate();
    std::cout << "Objective: " << std::scientific << result1.objective_value << std::endl;
    std::cout << "Residuals: [" << result1.residuals[0] << ", " << result1.residuals[1] << "]" << std::endl;
    std::cout << std::fixed;
    
    // Scenario 2: High weight on short-end
    std::cout << "\n--- Scenario 2: 10x Weight on Short-End ---" << std::endl;
    CurveOptimizer opt2;
    opt2.add(std::make_shared<OISDeposit>(0.5, 0.0280), 0.0, 10.0);  // 10x weight
    opt2.add(std::make_shared<OISDeposit>(5.0, 0.0380), 0.0, 1.0);
    
    auto result2 = opt2.calibrate();
    std::cout << "Objective: " << std::scientific << result2.objective_value << std::endl;
    std::cout << "Residuals: [" << result2.residuals[0] << ", " << result2.residuals[1] << "]" << std::endl;
    std::cout << std::fixed;
    
    std::cout << "\nNote: Higher weights lead to smaller residuals for those instruments." << std::endl;
}

int main() {
    std::cout << "\n";
    printSeparator();
    std::cout << "          CurveForge Optimization Library Examples" << std::endl;
    printSeparator();
    std::cout << "\n";
    
    try {
        example1_SimpleDeposits();
        std::cout << "\n\n";
        
        example2_MixedInstruments();
        std::cout << "\n\n";
        
        example3_FullCurveWithSwaps();
        std::cout << "\n\n";
        
        example4_WeightedCalibration();
        std::cout << "\n\n";
        
        printSeparator();
        std::cout << "All examples completed successfully!" << std::endl;
        printSeparator();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
