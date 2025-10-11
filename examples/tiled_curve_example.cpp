/**
 * @file tiled_curve_example.cpp
 * @brief Example demonstrating the use of TiledCurve for bond and swap pricing
 * 
 * This example shows how to:
 * 1. Calibrate a yield curve using market instruments
 * 2. Create a TiledCurve from the calibrated instantaneous forward rates
 * 3. Use the TiledCurve to price bonds and swaps
 */

#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/tiled_curve.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include "pricing/curve/irswap.hpp"
#include "pricing/bond/bond.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace pricing;
using namespace pricing::bond;

void printSeparator() {
    std::cout << std::string(70, '=') << std::endl;
}

void printHeader(const std::string& title) {
    printSeparator();
    std::cout << title << std::endl;
    printSeparator();
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    printHeader("TiledCurve Example: Bond and Swap Pricing");
    
    // ========================================================================
    // Step 1: Setup market instruments and calibrate curve
    // ========================================================================
    printHeader("Step 1: Calibrate Yield Curve from Market Instruments");
    
    CurveOptimizer::Config config;
    config.regularization_lambda = 0.001;  // Small regularization for smooth curves
    config.initial_forward_rate = 0.03;    // Initial guess
    
    CurveOptimizer optimizer(config);
    
    // Add OIS deposits (short-term instruments)
    std::cout << "Adding OIS Deposits:" << std::endl;
    auto deposit_3m = std::make_shared<OISDeposit>(0.25, 0.025);
    auto deposit_6m = std::make_shared<OISDeposit>(0.5, 0.026);
    auto deposit_1y = std::make_shared<OISDeposit>(1.0, 0.028);
    
    optimizer.add(deposit_3m, 0.0, 1.0);
    optimizer.add(deposit_6m, 0.0, 1.0);
    optimizer.add(deposit_1y, 0.0, 1.0);
    
    std::cout << "  3M @ 2.50%" << std::endl;
    std::cout << "  6M @ 2.60%" << std::endl;
    std::cout << "  1Y @ 2.80%" << std::endl;
    
    // Add Interest Rate Swaps (medium to long-term instruments)
    std::cout << "\nAdding Interest Rate Swaps:" << std::endl;
    
    // 2-year swap at 3.0%
    std::vector<double> swap_2y_times;
    for (int i = 1; i <= 4; ++i) swap_2y_times.push_back(0.5 * i);
    auto swap_2y = std::make_shared<IRSwap>(swap_2y_times, 0.030);
    optimizer.add(swap_2y, 0.0, 1.0);
    std::cout << "  2Y @ 3.00%" << std::endl;
    
    // 5-year swap at 3.5%
    std::vector<double> swap_5y_times;
    for (int i = 1; i <= 10; ++i) swap_5y_times.push_back(0.5 * i);
    auto swap_5y = std::make_shared<IRSwap>(swap_5y_times, 0.035);
    optimizer.add(swap_5y, 0.0, 1.0);
    std::cout << "  5Y @ 3.50%" << std::endl;
    
    // 10-year swap at 4.0%
    std::vector<double> swap_10y_times;
    for (int i = 1; i <= 20; ++i) swap_10y_times.push_back(0.5 * i);
    auto swap_10y = std::make_shared<IRSwap>(swap_10y_times, 0.040);
    optimizer.add(swap_10y, 0.0, 1.0);
    std::cout << "  10Y @ 4.00%" << std::endl;
    
    std::cout << "\nCalibrating curve..." << std::endl;
    auto result = optimizer.calibrate();
    
    std::cout << "Calibration " << (result.success ? "SUCCEEDED" : "FAILED") << std::endl;
    std::cout << "Objective value: " << std::scientific << result.objective_value << std::fixed << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    
    if (!result.success) {
        std::cerr << "Calibration failed!" << std::endl;
        return 1;
    }
    
    // ========================================================================
    // Step 2: Create TiledCurve from calibrated forward rates
    // ========================================================================
    printHeader("Step 2: Create TiledCurve from Calibrated Forward Rates");
    
    TiledCurve tiled_curve(result.pillar_times, result.forward_rates);
    
    std::cout << "Instantaneous Forward Rate Structure:" << std::endl;
    std::cout << std::setw(10) << "Time (Y)" << std::setw(15) << "Forward Rate" 
              << std::setw(18) << "Discount Factor" << std::endl;
    std::cout << std::string(43, '-') << std::endl;
    
    for (size_t i = 0; i < result.pillar_times.size(); ++i) {
        double t = result.pillar_times[i];
        double f = result.forward_rates[i];
        double df = tiled_curve.discount(t);
        std::cout << std::setw(10) << t 
                  << std::setw(15) << (f * 100.0) << "%" 
                  << std::setw(18) << df << std::endl;
    }
    
    // ========================================================================
    // Step 3: Price bonds using TiledCurve
    // ========================================================================
    printHeader("Step 3: Price Bonds using TiledCurve");
    
    std::cout << "Pricing bonds with different characteristics:\n" << std::endl;
    
    // Define discount function from TiledCurve
    auto discount_fn = [&tiled_curve](double t) { return tiled_curve.discount(t); };
    
    // Bond 1: 2-year, 3% coupon, semi-annual
    Bond bond_2y(100.0, 0.03, 2.0, 2);
    double price_2y = bond_2y.priceFromCurve(discount_fn);
    std::cout << "Bond 1 (2Y, 3% coupon, semi-annual):" << std::endl;
    std::cout << "  Price: " << std::setw(10) << price_2y << std::endl;
    std::cout << "  Clean Price: " << std::setw(10) << price_2y 
              << " (per 100 face value)" << std::endl;
    
    // Bond 2: 5-year, 4% coupon, semi-annual
    Bond bond_5y(100.0, 0.04, 5.0, 2);
    double price_5y = bond_5y.priceFromCurve(discount_fn);
    std::cout << "\nBond 2 (5Y, 4% coupon, semi-annual):" << std::endl;
    std::cout << "  Price: " << std::setw(10) << price_5y << std::endl;
    
    // Bond 3: 10-year, 5% coupon, semi-annual
    Bond bond_10y(100.0, 0.05, 10.0, 2);
    double price_10y = bond_10y.priceFromCurve(discount_fn);
    std::cout << "\nBond 3 (10Y, 5% coupon, semi-annual):" << std::endl;
    std::cout << "  Price: " << std::setw(10) << price_10y << std::endl;
    
    // ========================================================================
    // Step 4: Compare with traditional yield-based pricing
    // ========================================================================
    printHeader("Step 4: Compare Curve-Based vs Yield-Based Pricing");
    
    std::cout << "For the 5-year bond (4% coupon):\n" << std::endl;
    
    // Price using curve
    std::cout << "Curve-based price:  " << std::setw(10) << price_5y << std::endl;
    
    // Price using yield (approximate yield from discount factor)
    double df_5y = tiled_curve.discount(5.0);
    double approx_yield = -std::log(df_5y) / 5.0;  // Continuous compounding
    double price_from_yield = bond_5y.priceFromYield(approx_yield);
    
    std::cout << "Approximate yield:  " << std::setw(10) << (approx_yield * 100.0) << "%" << std::endl;
    std::cout << "Yield-based price:  " << std::setw(10) << price_from_yield << std::endl;
    std::cout << "Difference:         " << std::setw(10) << (price_5y - price_from_yield) << std::endl;
    
    std::cout << "\nNote: Small differences are due to different compounding conventions" << std::endl;
    std::cout << "(curve uses continuous compounding, yield method uses discrete)." << std::endl;
    
    // ========================================================================
    // Step 5: Verify swap pricing consistency
    // ========================================================================
    printHeader("Step 5: Verify Swap Pricing Consistency");
    
    std::cout << "Checking that calibrated swaps are correctly priced:\n" << std::endl;
    
    auto tiled_discount = [&tiled_curve](double t) { return tiled_curve.discount(t); };
    
    // Price the 2-year swap
    double solved_df_2y = swap_2y->solveDiscount(tiled_discount);
    double curve_df_2y = tiled_curve.discount(2.0);
    std::cout << "2Y Swap:" << std::endl;
    std::cout << "  Solved DF:  " << solved_df_2y << std::endl;
    std::cout << "  Curve DF:   " << curve_df_2y << std::endl;
    std::cout << "  Residual:   " << (curve_df_2y - solved_df_2y) << std::endl;
    
    // Price the 5-year swap
    double solved_df_5y = swap_5y->solveDiscount(tiled_discount);
    double curve_df_5y_maturity = tiled_curve.discount(5.0);
    std::cout << "\n5Y Swap:" << std::endl;
    std::cout << "  Solved DF:  " << solved_df_5y << std::endl;
    std::cout << "  Curve DF:   " << curve_df_5y_maturity << std::endl;
    std::cout << "  Residual:   " << (curve_df_5y_maturity - solved_df_5y) << std::endl;
    
    // ========================================================================
    // Step 6: Forward rate analysis
    // ========================================================================
    printHeader("Step 6: Forward Rate Analysis");
    
    std::cout << "Forward rates for different periods:\n" << std::endl;
    std::cout << std::setw(15) << "Period" 
              << std::setw(18) << "Forward Rate (%)" << std::endl;
    std::cout << std::string(33, '-') << std::endl;
    
    // 1-year forward rates
    std::vector<double> analysis_times = {0.0, 1.0, 2.0, 5.0, 7.0};
    for (size_t i = 0; i < analysis_times.size() - 1; ++i) {
        double t1 = analysis_times[i];
        double t2 = analysis_times[i + 1];
        double fwd_rate = tiled_curve.get_forward(t1, t2 - t1);
        double implied_rate = (1.0 / fwd_rate - 1.0) / (t2 - t1);
        
        std::cout << std::setw(4) << t1 << "Y to " << std::setw(4) << t2 << "Y"
                  << std::setw(18) << (implied_rate * 100.0) << "%" << std::endl;
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    printHeader("Summary");
    
    std::cout << "This example demonstrated:" << std::endl;
    std::cout << "1. Calibrating a yield curve from market instruments (deposits & swaps)" << std::endl;
    std::cout << "2. Creating a TiledCurve that stores instantaneous forward rates" << std::endl;
    std::cout << "3. Pricing bonds using the TiledCurve discount function" << std::endl;
    std::cout << "4. Comparing curve-based and yield-based pricing" << std::endl;
    std::cout << "5. Verifying swap pricing consistency" << std::endl;
    std::cout << "6. Analyzing forward rate structure" << std::endl;
    
    printSeparator();
    std::cout << "Example completed successfully!" << std::endl;
    
    return 0;
}
