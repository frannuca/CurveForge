/**
 * @file regularization_demo.cpp
 * @brief Demonstrates curve regularization for smooth forward curves
 */

#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace pricing;

void printForwardRates(const std::string& title, const YieldCurve& curve, 
                       const std::vector<double>& times) {
    std::cout << "\n" << title << std::endl;
    std::cout << std::setw(12) << "Time (Y)" 
              << std::setw(15) << "Discount"
              << std::setw(18) << "Inst. Forward (%)"
              << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (size_t i = 0; i < times.size(); ++i) {
        double t = times[i];
        double df = curve.discount(t);
        
        // Calculate instantaneous forward rate
        double dt = 0.001;  // Small time step
        double df_plus = curve.discount(t + dt);
        double fwd = -(std::log(df_plus) - std::log(df)) / dt * 100.0;
        
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << t
                  << std::setw(15) << std::setprecision(6) << df
                  << std::setw(18) << std::setprecision(4) << fwd
                  << std::endl;
    }
}

int main() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "        CurveForge: Regularization Demonstration" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Market data: deposits at various maturities
    struct MarketData {
        double tenor;
        double rate;
    };
    
    std::vector<MarketData> deposits = {
        {0.25, 0.0250},
        {0.50, 0.0280},
        {1.00, 0.0300},
        {2.00, 0.0320},
        {3.00, 0.0340},
        {5.00, 0.0380},
    };
    
    std::cout << "\nMarket Data:" << std::endl;
    std::cout << std::setw(12) << "Tenor (Y)" << std::setw(15) << "Rate (%)" << std::endl;
    std::cout << std::string(27, '-') << std::endl;
    for (const auto& data : deposits) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << data.tenor
                  << std::setw(15) << std::setprecision(4) << (data.rate * 100.0) << std::endl;
    }
    
    // Display times for forward curves (within calibrated range)
    std::vector<double> display_times;
    for (double t = 0.25; t <= 4.75; t += 0.25) {
        display_times.push_back(t);
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SCENARIO 1: No Regularization (Pure Data Fitting)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    {
        CurveOptimizer::Config config;
        config.regularization_lambda = 0.0;  // No regularization
        CurveOptimizer optimizer(config);
        
        for (const auto& data : deposits) {
            optimizer.add(std::make_shared<OISDeposit>(data.tenor, data.rate), 0.0, 1.0);
        }
        
        auto result = optimizer.calibrate();
        
        std::cout << "\nCalibration Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Objective Value: " << std::scientific << result.objective_value << std::fixed << std::endl;
        
        printForwardRates("Forward Curve (No Regularization):", result.curve, display_times);
        
        std::cout << "\nNote: Forward rates may show oscillations between pillars" << std::endl;
    }
    
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "SCENARIO 2: First-Order Regularization (Smooth Forward Rates)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    {
        CurveOptimizer::Config config;
        config.regularization_lambda = 0.01;   // Light regularization
        config.regularization_order = 1;        // First-order (penalize changes in rates)
        CurveOptimizer optimizer(config);
        
        for (const auto& data : deposits) {
            optimizer.add(std::make_shared<OISDeposit>(data.tenor, data.rate), 0.0, 1.0);
        }
        
        auto result = optimizer.calibrate();
        
        std::cout << "\nCalibration Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Objective Value: " << std::scientific << result.objective_value << std::fixed << std::endl;
        std::cout << "Regularization: lambda=" << config.regularization_lambda 
                  << ", order=" << config.regularization_order << std::endl;
        
        printForwardRates("Forward Curve (First-Order Regularization):", result.curve, display_times);
        
        std::cout << "\nNote: Forward rates show smoother transitions" << std::endl;
    }
    
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "SCENARIO 3: Second-Order Regularization (Smooth Curvature)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    {
        CurveOptimizer::Config config;
        config.regularization_lambda = 0.01;   // Light regularization
        config.regularization_order = 2;        // Second-order (penalize curvature changes)
        CurveOptimizer optimizer(config);
        
        for (const auto& data : deposits) {
            optimizer.add(std::make_shared<OISDeposit>(data.tenor, data.rate), 0.0, 1.0);
        }
        
        auto result = optimizer.calibrate();
        
        std::cout << "\nCalibration Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Objective Value: " << std::scientific << result.objective_value << std::fixed << std::endl;
        std::cout << "Regularization: lambda=" << config.regularization_lambda 
                  << ", order=" << config.regularization_order << " (RECOMMENDED)" << std::endl;
        
        printForwardRates("Forward Curve (Second-Order Regularization):", result.curve, display_times);
        
        std::cout << "\nNote: Smoothest curvature - best for derivative pricing" << std::endl;
    }
    
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "SCENARIO 4: Strong Regularization (Very Smooth)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    {
        CurveOptimizer::Config config;
        config.regularization_lambda = 0.10;   // Strong regularization
        config.regularization_order = 2;
        CurveOptimizer optimizer(config);
        
        for (const auto& data : deposits) {
            optimizer.add(std::make_shared<OISDeposit>(data.tenor, data.rate), 0.0, 1.0);
        }
        
        auto result = optimizer.calibrate();
        
        std::cout << "\nCalibration Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Objective Value: " << std::scientific << result.objective_value << std::fixed << std::endl;
        std::cout << "Regularization: lambda=" << config.regularization_lambda 
                  << ", order=" << config.regularization_order << std::endl;
        
        printForwardRates("Forward Curve (Strong Regularization):", result.curve, display_times);
        
        std::cout << "\nNote: Very smooth but may sacrifice some fitting accuracy" << std::endl;
    }
    
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\nRegularization Controls the Trade-off Between:" << std::endl;
    std::cout << "  - Data Fitting Accuracy (lower lambda = better fit)" << std::endl;
    std::cout << "  - Curve Smoothness (higher lambda = smoother curve)" << std::endl;
    std::cout << "\nRecommendations:" << std::endl;
    std::cout << "  - Use lambda=0.01, order=2 for most applications (default)" << std::endl;
    std::cout << "  - Increase lambda for noisy market data" << std::endl;
    std::cout << "  - Decrease lambda when exact fitting is required" << std::endl;
    std::cout << "  - Second-order regularization is preferred for smooth curvature" << std::endl;
    std::cout << "\n" << std::string(80, '=') << std::endl;
    
    return 0;
}
