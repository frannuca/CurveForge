#include "pricing/curve/tiled_curve.hpp"
#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include "pricing/curve/irswap.hpp"
#include "pricing/bond/bond.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace pricing;
using namespace pricing::bond;

// Helper function to check approximate equality
bool approxEqual(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// Test 1: Basic TiledCurve construction and discount factors
void testTiledCurveBasic() {
    std::cout << "\n=== Test 1: Basic TiledCurve Construction ===" << std::endl;
    
    // Create a simple tiled curve with constant forward rate
    std::vector<double> pillars = {1.0, 2.0, 3.0, 5.0, 10.0};
    std::vector<double> forwards = {0.03, 0.03, 0.03, 0.03, 0.03};
    
    TiledCurve curve(pillars, forwards);
    
    // Test discount factor at 1 year
    double df_1y = curve.discount(1.0);
    double expected_df_1y = std::exp(-0.03 * 1.0);
    std::cout << "DF at 1Y: " << df_1y << " (expected: " << expected_df_1y << ")" << std::endl;
    assert(approxEqual(df_1y, expected_df_1y, 1e-5));
    
    // Test discount factor at 5 years
    double df_5y = curve.discount(5.0);
    double expected_df_5y = std::exp(-0.03 * 5.0);
    std::cout << "DF at 5Y: " << df_5y << " (expected: " << expected_df_5y << ")" << std::endl;
    assert(approxEqual(df_5y, expected_df_5y, 1e-5));
    
    // Test instantaneous forward rate
    double f_2y = curve.instantaneous_forward(2.0);
    std::cout << "Instantaneous forward at 2Y: " << f_2y << " (expected: 0.03)" << std::endl;
    assert(approxEqual(f_2y, 0.03, 1e-5));
    
    std::cout << "✓ Test 1 passed" << std::endl;
}

// Test 2: TiledCurve with non-constant forward rates
void testTiledCurveNonConstant() {
    std::cout << "\n=== Test 2: TiledCurve with Non-Constant Forwards ===" << std::endl;
    
    // Create a curve with increasing forward rates
    std::vector<double> pillars = {1.0, 2.0, 5.0, 10.0};
    std::vector<double> forwards = {0.02, 0.03, 0.04, 0.05};
    
    TiledCurve curve(pillars, forwards);
    
    // Test that forwards are interpolated correctly
    double f_1_5y = curve.instantaneous_forward(1.5);
    double expected_f = 0.02 + 0.5 * (0.03 - 0.02); // Linear interpolation
    std::cout << "Interpolated forward at 1.5Y: " << f_1_5y << " (expected: " << expected_f << ")" << std::endl;
    assert(approxEqual(f_1_5y, expected_f, 1e-5));
    
    // Test discount factors
    double df_2y = curve.discount(2.0);
    std::cout << "DF at 2Y: " << df_2y << std::endl;
    assert(df_2y > 0.0 && df_2y < 1.0);
    
    // Test that discount factors are decreasing
    double df_1y = curve.discount(1.0);
    double df_5y = curve.discount(5.0);
    assert(df_1y > df_2y);
    assert(df_2y > df_5y);
    std::cout << "DF ordering correct: " << df_1y << " > " << df_2y << " > " << df_5y << std::endl;
    
    std::cout << "✓ Test 2 passed" << std::endl;
}

// Test 3: TiledCurve from optimizer calibration
void testTiledCurveFromOptimizer() {
    std::cout << "\n=== Test 3: TiledCurve from Optimizer Calibration ===" << std::endl;
    
    CurveOptimizer optimizer;
    
    // Add market instruments
    auto deposit_1y = std::make_shared<OISDeposit>(1.0, 0.03);
    auto deposit_2y = std::make_shared<OISDeposit>(2.0, 0.035);
    auto deposit_5y = std::make_shared<OISDeposit>(5.0, 0.04);
    
    optimizer.add(deposit_1y, 0.0, 1.0);
    optimizer.add(deposit_2y, 0.0, 1.0);
    optimizer.add(deposit_5y, 0.0, 1.0);
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Calibration success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << result.objective_value << std::endl;
    assert(result.success);
    
    // Create TiledCurve from calibration result
    TiledCurve tiled_curve(result.pillar_times, result.forward_rates);
    
    // Compare discount factors between YieldCurve and TiledCurve
    std::vector<double> test_times = {1.0, 2.0, 3.0, 5.0};
    for (double t : test_times) {
        double df_yield = result.curve.discount(t);
        double df_tiled = tiled_curve.discount(t);
        std::cout << "DF at " << t << "Y: YieldCurve=" << df_yield 
                  << ", TiledCurve=" << df_tiled << std::endl;
        assert(approxEqual(df_yield, df_tiled, 1e-4));
    }
    
    std::cout << "✓ Test 3 passed" << std::endl;
}

// Test 4: Bond pricing with TiledCurve
void testBondPricingWithTiledCurve() {
    std::cout << "\n=== Test 4: Bond Pricing with TiledCurve ===" << std::endl;
    
    // Create a bond: 5% coupon, 100 face value, 5 year maturity, semi-annual
    Bond bond(100.0, 0.05, 5.0, 2);
    
    // Create a flat curve at 4% yield
    std::vector<double> pillars = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> forwards(pillars.size(), 0.04);
    
    TiledCurve curve(pillars, forwards);
    
    // Price bond using TiledCurve
    auto discount_fn = [&curve](double t) { return curve.discount(t); };
    double price_from_curve = bond.priceFromCurve(discount_fn);
    
    // Price bond using yield method (4% yield)
    double price_from_yield = bond.priceFromYield(0.04);
    
    std::cout << "Price from curve: " << price_from_curve << std::endl;
    std::cout << "Price from yield: " << price_from_yield << std::endl;
    
    // The prices should be similar (not exact due to different compounding conventions)
    // TiledCurve uses continuous compounding, yield method uses discrete
    assert(std::abs(price_from_curve - price_from_yield) < 5.0);
    assert(price_from_curve > 100.0); // Bond should trade at premium (coupon > yield)
    
    std::cout << "✓ Test 4 passed" << std::endl;
}

// Test 5: Swap pricing consistency with TiledCurve
void testSwapPricingWithTiledCurve() {
    std::cout << "\n=== Test 5: Swap Pricing with TiledCurve ===" << std::endl;
    
    // Create calibrated curve using swap instruments
    CurveOptimizer optimizer;
    
    std::vector<double> swap_times_2y = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> swap_times_5y = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    
    auto swap_2y = std::make_shared<IRSwap>(swap_times_2y, 0.035);
    auto swap_5y = std::make_shared<IRSwap>(swap_times_5y, 0.04);
    
    optimizer.add(swap_2y, 0.0, 1.0);
    optimizer.add(swap_5y, 0.0, 1.0);
    
    auto result = optimizer.calibrate();
    
    std::cout << "Calibration success: " << (result.success ? "YES" : "NO") << std::endl;
    assert(result.success);
    
    // Create TiledCurve
    TiledCurve tiled_curve(result.pillar_times, result.forward_rates);
    
    // Verify that swaps can be priced correctly with TiledCurve
    auto tiled_discount_fn = [&tiled_curve](double t) { return tiled_curve.discount(t); };
    auto yield_discount_fn = [&result](double t) { return result.curve.discount(t); };
    
    double solved_df_2y_tiled = swap_2y->solveDiscount(tiled_discount_fn);
    double solved_df_2y_yield = swap_2y->solveDiscount(yield_discount_fn);
    
    std::cout << "Swap 2Y solved DF (TiledCurve): " << solved_df_2y_tiled << std::endl;
    std::cout << "Swap 2Y solved DF (YieldCurve): " << solved_df_2y_yield << std::endl;
    assert(approxEqual(solved_df_2y_tiled, solved_df_2y_yield, 1e-4));
    
    std::cout << "✓ Test 5 passed" << std::endl;
}

// Test 6: Integration test - Full calibration and pricing workflow
void testFullWorkflow() {
    std::cout << "\n=== Test 6: Full Calibration and Pricing Workflow ===" << std::endl;
    
    // Step 1: Calibrate curve using market instruments
    CurveOptimizer::Config config;
    config.regularization_lambda = 0.001;
    CurveOptimizer optimizer(config);
    
    // Add deposits
    optimizer.add(std::make_shared<OISDeposit>(1.0, 0.025), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(2.0, 0.030), 0.0, 1.0);
    
    // Add swaps
    std::vector<double> swap_5y_times = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    std::vector<double> swap_10y_times;
    for (int i = 1; i <= 20; ++i) {
        swap_10y_times.push_back(0.5 * i);
    }
    
    optimizer.add(std::make_shared<IRSwap>(swap_5y_times, 0.035), 0.0, 1.0);
    optimizer.add(std::make_shared<IRSwap>(swap_10y_times, 0.04), 0.0, 1.0);
    
    auto result = optimizer.calibrate();
    
    std::cout << "Calibration success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << result.objective_value << std::endl;
    assert(result.success);
    
    // Step 2: Create TiledCurve from calibration
    TiledCurve tiled_curve(result.pillar_times, result.forward_rates);
    
    // Step 3: Price a bond using the calibrated curve
    Bond bond(100.0, 0.04, 10.0, 2);
    auto discount_fn = [&tiled_curve](double t) { return tiled_curve.discount(t); };
    double bond_price = bond.priceFromCurve(discount_fn);
    
    std::cout << "Bond price: " << bond_price << std::endl;
    assert(bond_price > 0.0 && bond_price < 200.0);
    
    // Step 4: Verify forward rate structure
    std::cout << "Forward rate structure:" << std::endl;
    for (size_t i = 0; i < result.pillar_times.size(); ++i) {
        std::cout << "  t=" << std::setw(6) << result.pillar_times[i] 
                  << ", f=" << std::setw(8) << result.forward_rates[i] << std::endl;
    }
    
    std::cout << "✓ Test 6 passed" << std::endl;
}

int main() {
    std::cout << "Running TiledCurve Integration Tests" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        testTiledCurveBasic();
        testTiledCurveNonConstant();
        testTiledCurveFromOptimizer();
        testBondPricingWithTiledCurve();
        testSwapPricingWithTiledCurve();
        testFullWorkflow();
        
        std::cout << "\n=====================================" << std::endl;
        std::cout << "All tests passed successfully!" << std::endl;
        std::cout << "TILED_CURVE_OK" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
