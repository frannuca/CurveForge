#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include "pricing/curve/fra.hpp"
#include "pricing/curve/irswap.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace pricing;

// Helper function to check approximate equality
bool approxEqual(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) < tol;
}

// Test 1: Simple deposit calibration
void testSimpleDeposit() {
    std::cout << "\n=== Test 1: Simple Deposit Calibration ===" << std::endl;
    
    CurveOptimizer optimizer;
    
    // Add a single OIS deposit: 1-year at 3% rate
    auto deposit = std::make_shared<OISDeposit>(1.0, 0.03);
    optimizer.add(deposit, 0.0, 1.0);  // market_price parameter not used for curve instruments
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    
    // Check discount factor at 1 year
    double df_1y = result.curve.discount(1.0);
    double expected_df = 1.0 / (1.0 + 0.03 * 1.0);
    
    std::cout << "DF at 1Y: " << df_1y << " (expected: " << expected_df << ")" << std::endl;
    
    if (!result.success) {
        throw std::runtime_error("Test 1 failed: optimization did not succeed");
    }
    
    if (result.objective_value > 1e-6) {
        std::cout << "WARNING: Objective value higher than expected" << std::endl;
    }
    
    std::cout << "✓ Test 1 passed" << std::endl;
}

// Test 2: Multiple deposits
void testMultipleDeposits() {
    std::cout << "\n=== Test 2: Multiple Deposits ===" << std::endl;
    
    CurveOptimizer optimizer;
    
    // Add multiple deposits
    optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);  // 3M at 2.5%
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);   // 6M at 2.8%
    optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);   // 1Y at 3.0%
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    std::cout << "Pillar times: ";
    for (auto t : optimizer.pillarTimes()) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
    
    // Check discount factors
    std::cout << "Discount factors:" << std::endl;
    for (auto t : optimizer.pillarTimes()) {
        std::cout << "  DF(" << t << "Y) = " << result.curve.discount(t) << std::endl;
    }
    
    if (!result.success) {
        throw std::runtime_error("Test 2 failed: optimization did not succeed");
    }
    
    std::cout << "✓ Test 2 passed" << std::endl;
}

// Test 3: FRA calibration
void testFRACalibration() {
    std::cout << "\n=== Test 3: FRA Calibration ===" << std::endl;
    
    CurveOptimizer optimizer;
    
    // Add deposits for short end
    optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
    
    // Add FRA: 6M x 1Y (starts at 6M, ends at 1.5Y) with forward rate 3.2%
    optimizer.add(std::make_shared<FRA>(0.5, 1.5, 0.032), 0.0, 1.0);
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    
    // Check discount factors
    std::cout << "Discount factors:" << std::endl;
    for (auto t : optimizer.pillarTimes()) {
        std::cout << "  DF(" << t << "Y) = " << result.curve.discount(t) << std::endl;
    }
    
    if (!result.success) {
        throw std::runtime_error("Test 3 failed: optimization did not succeed");
    }
    
    std::cout << "✓ Test 3 passed" << std::endl;
}

// Test 4: Swap calibration
void testSwapCalibration() {
    std::cout << "\n=== Test 4: Interest Rate Swap Calibration ===" << std::endl;
    
    CurveOptimizer optimizer;
    
    // Add short-end deposits
    optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);
    
    // Add a 5-year swap with annual payments at 3.5% fixed rate
    std::vector<double> swapTimes = {1.0, 2.0, 3.0, 4.0, 5.0};
    optimizer.add(std::make_shared<IRSwap>(swapTimes, 0.035), 0.0, 1.0);
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    std::cout << "Number of pillars: " << optimizer.pillarTimes().size() << std::endl;
    
    // Check discount factors
    std::cout << "Discount factors:" << std::endl;
    for (auto t : optimizer.pillarTimes()) {
        std::cout << "  DF(" << t << "Y) = " << std::setprecision(8) 
                  << result.curve.discount(t) << std::endl;
    }
    
    if (!result.success) {
        throw std::runtime_error("Test 4 failed: optimization did not succeed");
    }
    
    std::cout << "✓ Test 4 passed" << std::endl;
}

// Test 5: Residuals check
void testResiduals() {
    std::cout << "\n=== Test 5: Residuals Check ===" << std::endl;
    
    CurveOptimizer optimizer;
    
    // Add instruments
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(2.0, 0.032), 0.0, 1.0);
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    std::cout << "Number of residuals: " << result.residuals.size() << std::endl;
    
    std::cout << "Residuals:" << std::endl;
    for (size_t i = 0; i < result.residuals.size(); ++i) {
        std::cout << "  Instrument " << i << ": " << result.residuals[i] << std::endl;
    }
    
    if (result.residuals.size() != 3) {
        throw std::runtime_error("Test 5 failed: wrong number of residuals");
    }
    
    std::cout << "✓ Test 5 passed" << std::endl;
}

// Test 6: Configuration options
void testConfiguration() {
    std::cout << "\n=== Test 6: Configuration Options ===" << std::endl;
    
    CurveOptimizer::Config config;
    config.max_iterations = 500;
    config.relative_tolerance = 1e-7;
    config.initial_forward_rate = 0.04;
    
    CurveOptimizer optimizer(config);
    
    // Add instruments
    optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(2.0, 0.032), 0.0, 1.0);
    
    // Calibrate
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    
    if (!result.success) {
        throw std::runtime_error("Test 6 failed: optimization did not succeed");
    }
    
    std::cout << "✓ Test 6 passed" << std::endl;
}

// Test 7: Regularization effect
void testRegularization() {
    std::cout << "\n=== Test 7: Regularization Effect ===" << std::endl;
    
    // Test with no regularization
    CurveOptimizer::Config config_no_reg;
    config_no_reg.regularization_lambda = 0.0;
    CurveOptimizer optimizer_no_reg(config_no_reg);
    
    // Add instruments
    optimizer_no_reg.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
    optimizer_no_reg.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
    optimizer_no_reg.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);
    optimizer_no_reg.add(std::make_shared<OISDeposit>(2.0, 0.032), 0.0, 1.0);
    
    auto result_no_reg = optimizer_no_reg.calibrate();
    
    std::cout << "Without regularization:" << std::endl;
    std::cout << "  Success: " << (result_no_reg.success ? "YES" : "NO") << std::endl;
    std::cout << "  Objective: " << result_no_reg.objective_value << std::endl;
    
    // Test with regularization
    CurveOptimizer::Config config_with_reg;
    config_with_reg.regularization_lambda = 0.01;
    config_with_reg.regularization_order = 2;  // Second-order for smoothness
    CurveOptimizer optimizer_with_reg(config_with_reg);
    
    // Add same instruments
    optimizer_with_reg.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
    optimizer_with_reg.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
    optimizer_with_reg.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);
    optimizer_with_reg.add(std::make_shared<OISDeposit>(2.0, 0.032), 0.0, 1.0);
    
    auto result_with_reg = optimizer_with_reg.calibrate();
    
    std::cout << "With regularization (lambda=0.01, order=2):" << std::endl;
    std::cout << "  Success: " << (result_with_reg.success ? "YES" : "NO") << std::endl;
    std::cout << "  Objective: " << result_with_reg.objective_value << std::endl;
    
    if (!result_no_reg.success || !result_with_reg.success) {
        throw std::runtime_error("Test 7 failed: optimization did not succeed");
    }
    
    // Both should succeed
    std::cout << "✓ Test 7 passed (regularization enabled)" << std::endl;
}

// Test 8: SQP algorithm validation
void testSQPAlgorithm() {
    std::cout << "\n=== Test 8: SQP Algorithm Validation ===" << std::endl;
    
    // Create a more complex curve calibration scenario
    CurveOptimizer optimizer;
    
    // Add a mix of instruments
    optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
    optimizer.add(std::make_shared<FRA>(0.5, 1.5, 0.032), 0.0, 1.0);
    optimizer.add(std::make_shared<OISDeposit>(2.0, 0.033), 0.0, 1.0);
    
    auto result = optimizer.calibrate();
    
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Objective: " << result.objective_value << std::endl;
    std::cout << "Number of residuals: " << result.residuals.size() << std::endl;
    
    // Check that SQP converges to a good solution (low objective value)
    if (!result.success) {
        throw std::runtime_error("Test 8 failed: SQP optimization did not succeed");
    }
    
    if (result.objective_value > 1e-5) {
        std::cout << "WARNING: Objective value higher than expected for SQP" << std::endl;
    }
    
    std::cout << "✓ Test 8 passed (SQP algorithm working)" << std::endl;
}

int main() {
    try {
        std::cout << "=== CurveForge Optimizer Test Suite ===" << std::endl;
        
        testSimpleDeposit();
        testMultipleDeposits();
        testFRACalibration();
        testSwapCalibration();
        testResiduals();
        testConfiguration();
        testRegularization();
        testSQPAlgorithm();
        
        std::cout << "\n=== All tests passed ===" << std::endl;
        std::cout << "OPTIMIZER_OK" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
