/**
 * Bond Pricing Module - Comprehensive Examples
 * 
 * This file demonstrates all features of the bond pricing module with
 * practical examples and use cases.
 */

#include "pricing/bond.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace pricing::bond;

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void example1_basicBondPricing() {
    printHeader("Example 1: Basic Bond Pricing");
    
    // Create a 5% coupon bond, $100 face value, 10 years maturity, semi-annual
    Bond bond(100.0, 0.05, 10.0, 2);
    
    std::cout << "\nBond Characteristics:" << std::endl;
    std::cout << "  Face Value: $" << bond.faceValue() << std::endl;
    std::cout << "  Coupon Rate: " << (bond.couponRate() * 100) << "%" << std::endl;
    std::cout << "  Maturity: " << bond.maturity() << " years" << std::endl;
    std::cout << "  Payments per year: 2 (semi-annual)" << std::endl;
    
    // Price at different yields
    std::cout << "\nPricing at different yields:" << std::endl;
    for (double yield = 0.03; yield <= 0.07; yield += 0.01) {
        double price = bond.priceFromYield(yield);
        std::string type = (price > 100) ? "Premium" : 
                          (price < 100) ? "Discount" : "Par";
        std::cout << "  Yield " << (yield * 100) << "%: $" 
                  << std::fixed << std::setprecision(2) << price 
                  << " (" << type << ")" << std::endl;
    }
    
    // Round-trip: yield from price
    double target_price = 95.0;
    double implied_yield = bond.yieldFromPrice(target_price);
    std::cout << "\nImplied yield at $" << target_price << ": " 
              << (implied_yield * 100) << "%" << std::endl;
}

void example2_durationConvexity() {
    printHeader("Example 2: Duration and Convexity");
    
    Bond bond(100.0, 0.06, 5.0, 2);
    double yield = 0.06;
    
    double price = bond.priceFromYield(yield);
    double duration = bond.duration(yield);
    double mod_duration = bond.modifiedDuration(yield);
    double convexity = bond.convexity(yield);
    
    std::cout << "\nBond at " << (yield * 100) << "% yield:" << std::endl;
    std::cout << "  Price: $" << std::fixed << std::setprecision(2) << price << std::endl;
    std::cout << "  Macaulay Duration: " << std::setprecision(4) << duration << " years" << std::endl;
    std::cout << "  Modified Duration: " << mod_duration << std::endl;
    std::cout << "  Convexity: " << convexity << std::endl;
    
    // Estimate price change using duration and convexity
    std::cout << "\nPrice sensitivity analysis:" << std::endl;
    for (double delta_y = -0.01; delta_y <= 0.01; delta_y += 0.005) {
        if (delta_y == 0.0) continue;
        
        double new_price = bond.priceFromYield(yield + delta_y);
        double actual_change = new_price - price;
        
        // Duration approximation: ΔP ≈ -D_mod * Δy * P
        double duration_est = -mod_duration * delta_y * price;
        
        // Duration + Convexity: ΔP ≈ -D_mod * Δy * P + 0.5 * C * (Δy)^2 * P
        double convexity_est = duration_est + 0.5 * convexity * delta_y * delta_y * price;
        
        std::cout << "  Δy = " << std::setw(6) << (delta_y * 100) << "bp: "
                  << "Actual ΔP = $" << std::setw(6) << std::setprecision(2) << actual_change
                  << ", Duration est = $" << std::setw(6) << duration_est
                  << ", With convexity = $" << std::setw(6) << convexity_est << std::endl;
    }
}

void example3_carryRoll() {
    printHeader("Example 3: Carry and Roll Analysis");
    
    Bond bond(100.0, 0.05, 10.0, 2);
    
    std::cout << "\n6-month horizon analysis:" << std::endl;
    
    // Scenario 1: Yield unchanged
    CarryRollMetrics scenario1 = calculateCarryRoll(bond, 0.05, 0.05, 0.5);
    std::cout << "\nScenario 1: Yield unchanged at 5%" << std::endl;
    std::cout << "  Carry: $" << std::fixed << std::setprecision(2) << scenario1.carry << std::endl;
    std::cout << "  Roll: $" << scenario1.roll << std::endl;
    std::cout << "  Total Return: $" << scenario1.total_return << std::endl;
    
    // Scenario 2: Yield declines
    CarryRollMetrics scenario2 = calculateCarryRoll(bond, 0.05, 0.04, 0.5);
    std::cout << "\nScenario 2: Yield declines to 4%" << std::endl;
    std::cout << "  Carry: $" << scenario2.carry << std::endl;
    std::cout << "  Roll: $" << scenario2.roll << std::endl;
    std::cout << "  Total Return: $" << scenario2.total_return << std::endl;
    std::cout << "  Additional return from yield decline: $" 
              << (scenario2.total_return - scenario1.total_return) << std::endl;
    
    // Scenario 3: Yield rises
    CarryRollMetrics scenario3 = calculateCarryRoll(bond, 0.05, 0.06, 0.5);
    std::cout << "\nScenario 3: Yield rises to 6%" << std::endl;
    std::cout << "  Carry: $" << scenario3.carry << std::endl;
    std::cout << "  Roll: $" << scenario3.roll << std::endl;
    std::cout << "  Total Return: $" << scenario3.total_return << std::endl;
    std::cout << "  Loss from yield increase: $" 
              << (scenario3.total_return - scenario1.total_return) << std::endl;
}

void example4_bondFutures() {
    printHeader("Example 4: Bond Futures Pricing");
    
    // Create a basket of deliverable bonds
    Bond bond1(100.0, 0.06, 10.0, 2);
    Bond bond2(100.0, 0.05, 15.0, 2);
    Bond bond3(100.0, 0.055, 12.0, 2);
    
    std::vector<Bond> deliverable_bonds = {bond1, bond2, bond3};
    
    // Calculate conversion factors
    double notional_coupon = 0.06;
    std::vector<double> conversion_factors;
    
    std::cout << "\nDeliverable Bonds and Conversion Factors:" << std::endl;
    for (size_t i = 0; i < deliverable_bonds.size(); ++i) {
        double cf = calculateConversionFactor(deliverable_bonds[i], notional_coupon);
        conversion_factors.push_back(cf);
        std::cout << "  Bond " << (i + 1) << ": " 
                  << (deliverable_bonds[i].couponRate() * 100) << "% coupon, "
                  << deliverable_bonds[i].maturity() << "y maturity, CF = " 
                  << std::fixed << std::setprecision(4) << cf << std::endl;
    }
    
    // Create futures contract (6 months to delivery)
    BondFuture future(0.5, deliverable_bonds, conversion_factors);
    
    // Current bond prices
    std::vector<double> bond_prices = {105.0, 98.0, 101.5};
    double repo_rate = 0.03;
    
    std::cout << "\nMarket Prices:" << std::endl;
    for (size_t i = 0; i < bond_prices.size(); ++i) {
        std::cout << "  Bond " << (i + 1) << ": $" 
                  << std::setprecision(2) << bond_prices[i] << std::endl;
    }
    
    // Calculate futures price
    double futures_price = future.futuresPrice(bond_prices, repo_rate);
    std::cout << "\nTheoretical Futures Price: $" << futures_price << std::endl;
    std::cout << "Repo Rate: " << (repo_rate * 100) << "%" << std::endl;
    
    // Identify cheapest to deliver
    size_t ctd = future.cheapestToDeliver(bond_prices, repo_rate);
    std::cout << "\nCheapest to Deliver: Bond " << (ctd + 1) << std::endl;
    
    // Calculate implied repo rates
    std::cout << "\nImplied Repo Rates:" << std::endl;
    for (size_t i = 0; i < deliverable_bonds.size(); ++i) {
        double implied_repo = future.impliedRepoRate(i, bond_prices[i], futures_price);
        std::cout << "  Bond " << (i + 1) << ": " 
                  << std::setprecision(2) << (implied_repo * 100) << "%" << std::endl;
    }
}

void example5_INSSBonds() {
    printHeader("Example 5: INSS Bond Pricing (Brazilian Social Security Bonds)");
    
    // Create INSS bond with 15% tax on coupons
    INSSBond inss_bond(100.0, 0.05, 10.0, 2, 0.15, false);
    
    // Compare with regular bond
    Bond regular_bond(100.0, 0.05, 10.0, 2);
    
    double yield = 0.05;
    double inss_price = inss_bond.priceFromYield(yield);
    double regular_price = regular_bond.priceFromYield(yield);
    
    std::cout << "\nComparison at " << (yield * 100) << "% yield:" << std::endl;
    std::cout << "  INSS Bond Price: $" << std::fixed << std::setprecision(2) 
              << inss_price << std::endl;
    std::cout << "  Regular Bond Price: $" << regular_price << std::endl;
    std::cout << "  Price difference: $" << (regular_price - inss_price) 
              << " (due to 15% tax)" << std::endl;
    
    // Calculate comprehensive metrics
    INSSMetrics metrics = calculateINSSMetrics(inss_bond, inss_price);
    
    std::cout << "\nINSS Bond Metrics:" << std::endl;
    std::cout << "  Net (after-tax) yield: " << std::setprecision(2) 
              << (metrics.net_yield * 100) << "%" << std::endl;
    std::cout << "  Gross (pre-tax) yield: " 
              << (metrics.gross_yield * 100) << "%" << std::endl;
    std::cout << "  Present value of taxes: $" << metrics.tax_pv << std::endl;
    std::cout << "  Duration: " << std::setprecision(4) 
              << metrics.duration << " years" << std::endl;
    std::cout << "  Convexity: " << metrics.convexity << std::endl;
    
    // Show after-tax coupon calculation
    double gross_coupon = 2.5;
    double after_tax_coupon = inss_bond.afterTaxCoupon(gross_coupon);
    std::cout << "\nCoupon Calculation:" << std::endl;
    std::cout << "  Gross coupon: $" << std::setprecision(2) << gross_coupon << std::endl;
    std::cout << "  Tax (15%): $" << (gross_coupon * 0.15) << std::endl;
    std::cout << "  After-tax coupon: $" << after_tax_coupon << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║    Bond Pricing Module - Comprehensive Examples           ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        example1_basicBondPricing();
        example2_durationConvexity();
        example3_carryRoll();
        example4_bondFutures();
        example5_INSSBonds();
        
        printHeader("Examples Completed Successfully");
        std::cout << "\nAll examples executed without errors!" << std::endl;
        std::cout << "\nFor more information, see BOND_PRICING_README.md" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
