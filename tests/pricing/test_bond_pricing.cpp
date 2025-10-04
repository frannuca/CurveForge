#include "pricing/bond.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cassert>

using namespace pricing::bond;

// Helper function for floating point comparison
bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

void testBasicBondPricing() {
    std::cout << "Testing basic bond pricing..." << std::endl;
    
    // Create a 5% coupon bond, 100 face value, 10 year maturity, semi-annual payments
    Bond bond(100.0, 0.05, 10.0, 2);
    
    // Test 1: Price at par (yield = coupon rate)
    double price_at_par = bond.priceFromYield(0.05);
    assert(approxEqual(price_at_par, 100.0, 0.1));
    std::cout << "  ✓ Price at par yield: " << price_at_par << std::endl;
    
    // Test 2: Price when yield < coupon (premium bond)
    double price_premium = bond.priceFromYield(0.04);
    assert(price_premium > 100.0);
    std::cout << "  ✓ Price at 4% yield (premium): " << price_premium << std::endl;
    
    // Test 3: Price when yield > coupon (discount bond)
    double price_discount = bond.priceFromYield(0.06);
    assert(price_discount < 100.0);
    std::cout << "  ✓ Price at 6% yield (discount): " << price_discount << std::endl;
    
    // Test 4: Round-trip yield calculation
    double yield_back = bond.yieldFromPrice(price_premium);
    assert(approxEqual(yield_back, 0.04, 1e-5));
    std::cout << "  ✓ Yield from price (round-trip): " << yield_back << std::endl;
    
    std::cout << "✓ Basic bond pricing tests passed" << std::endl << std::endl;
}

void testDurationConvexity() {
    std::cout << "Testing duration and convexity..." << std::endl;
    
    // Create a bond
    Bond bond(100.0, 0.06, 5.0, 2);
    
    // Calculate duration at 6% yield
    double dur = bond.duration(0.06);
    double mod_dur = bond.modifiedDuration(0.06);
    double conv = bond.convexity(0.06);
    
    std::cout << "  Duration: " << dur << " years" << std::endl;
    std::cout << "  Modified Duration: " << mod_dur << " years" << std::endl;
    std::cout << "  Convexity: " << conv << std::endl;
    
    // Duration should be positive and less than maturity
    assert(dur > 0.0 && dur < 5.0);
    
    // Modified duration should be less than Macaulay duration for discrete compounding
    // Modified Duration = Macaulay Duration / (1 + y/freq)
    assert(mod_dur < dur);
    double expected_mod_dur = dur / (1.0 + 0.06 / 2.0);
    assert(approxEqual(mod_dur, expected_mod_dur, 1e-6));
    
    // Convexity should be positive
    assert(conv > 0.0);
    
    std::cout << "✓ Duration and convexity tests passed" << std::endl << std::endl;
}

void testAccruedInterest() {
    std::cout << "Testing accrued interest..." << std::endl;
    
    Bond bond(100.0, 0.04, 2.0, 2);  // 4% coupon, 2 years, semi-annual
    
    // Accrued interest at t=0 should be 0
    double ai_0 = bond.accruedInterest(0.0);
    assert(approxEqual(ai_0, 0.0, 1e-6));
    std::cout << "  ✓ Accrued interest at t=0: " << ai_0 << std::endl;
    
    // Accrued interest at t=0.25 (halfway to first coupon)
    double ai_quarter = bond.accruedInterest(0.25);
    assert(ai_quarter > 0.0);
    std::cout << "  ✓ Accrued interest at t=0.25: " << ai_quarter << std::endl;
    
    std::cout << "✓ Accrued interest tests passed" << std::endl << std::endl;
}

void testCarryRoll() {
    std::cout << "Testing carry and roll analysis..." << std::endl;
    
    Bond bond(100.0, 0.05, 10.0, 2);
    
    // Calculate carry and roll over 6 months (to include one coupon payment)
    CarryRollMetrics metrics = calculateCarryRoll(bond, 0.05, 0.05, 0.5);
    
    std::cout << "  Carry: " << metrics.carry << std::endl;
    std::cout << "  Roll: " << metrics.roll << std::endl;
    std::cout << "  Total Return: " << metrics.total_return << std::endl;
    
    // Carry should be positive (one coupon payment of 2.5)
    assert(metrics.carry > 0.0);
    assert(approxEqual(metrics.carry, 2.5, 0.1));
    
    // When yield is unchanged, roll should be small (just time decay)
    // Total return = carry + roll
    assert(approxEqual(metrics.total_return, metrics.carry + metrics.roll, 1e-6));
    
    std::cout << "✓ Carry and roll tests passed" << std::endl << std::endl;
}

void testBondFutures() {
    std::cout << "Testing bond futures pricing..." << std::endl;
    
    // Create deliverable bonds
    Bond bond1(100.0, 0.06, 10.0, 2);
    Bond bond2(100.0, 0.05, 15.0, 2);
    
    std::vector<Bond> deliverable_bonds = {bond1, bond2};
    std::vector<double> conversion_factors = {1.05, 0.98};
    
    BondFuture future(0.5, deliverable_bonds, conversion_factors);
    
    // Test futures pricing
    std::vector<double> bond_prices = {105.0, 98.0};
    double futures_price = future.futuresPrice(bond_prices, 0.03);
    
    std::cout << "  Futures price: " << futures_price << std::endl;
    assert(futures_price > 0.0);
    
    // Test implied repo rate
    double implied_repo = future.impliedRepoRate(0, bond_prices[0], futures_price);
    std::cout << "  Implied repo rate: " << implied_repo << std::endl;
    
    // Test cheapest to deliver
    size_t ctd = future.cheapestToDeliver(bond_prices, 0.03);
    std::cout << "  Cheapest to deliver index: " << ctd << std::endl;
    assert(ctd < deliverable_bonds.size());
    
    // Test conversion factor calculation
    double cf = calculateConversionFactor(bond1, 0.06);
    std::cout << "  Conversion factor: " << cf << std::endl;
    assert(cf > 0.0);
    
    std::cout << "✓ Bond futures tests passed" << std::endl << std::endl;
}

void testINSSBonds() {
    std::cout << "Testing INSS bond pricing..." << std::endl;
    
    // Create INSS bond with 15% tax rate
    INSSBond inss_bond(100.0, 0.05, 10.0, 2, 0.15, false);
    
    // Price should be lower than regular bond due to tax on coupons
    Bond regular_bond(100.0, 0.05, 10.0, 2);
    
    double inss_price = inss_bond.priceFromYield(0.05);
    double regular_price = regular_bond.priceFromYield(0.05);
    
    std::cout << "  INSS bond price: " << inss_price << std::endl;
    std::cout << "  Regular bond price: " << regular_price << std::endl;
    
    // INSS bond should be cheaper due to taxes
    assert(inss_price < regular_price);
    
    // Test after-tax coupon calculation
    double after_tax = inss_bond.afterTaxCoupon(2.5);
    assert(approxEqual(after_tax, 2.5 * 0.85, 1e-6));
    std::cout << "  ✓ After-tax coupon: " << after_tax << std::endl;
    
    // Test INSS metrics calculation
    INSSMetrics metrics = calculateINSSMetrics(inss_bond, inss_price);
    std::cout << "  Net yield: " << metrics.net_yield << std::endl;
    std::cout << "  Gross yield: " << metrics.gross_yield << std::endl;
    std::cout << "  Tax PV: " << metrics.tax_pv << std::endl;
    
    assert(metrics.net_yield > 0.0);
    assert(metrics.tax_pv > 0.0);
    
    std::cout << "✓ INSS bond tests passed" << std::endl << std::endl;
}

void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test zero-coupon bond
    Bond zero_coupon(100.0, 0.0, 5.0, 1);
    double zc_price = zero_coupon.priceFromYield(0.05);
    std::cout << "  ✓ Zero-coupon bond price: " << zc_price << std::endl;
    assert(zc_price > 0.0 && zc_price < 100.0);
    
    // Price should be approximately face_value * exp(-yield * maturity)
    double expected_zc = 100.0 * std::exp(-0.05 * 5.0);
    assert(approxEqual(zc_price, expected_zc, 1.0));
    
    // Test short maturity bond
    Bond short_bond(100.0, 0.03, 0.5, 2);
    double short_price = short_bond.priceFromYield(0.03);
    std::cout << "  ✓ Short maturity bond price: " << short_price << std::endl;
    assert(approxEqual(short_price, 100.0, 0.5));
    
    std::cout << "✓ Edge case tests passed" << std::endl << std::endl;
}

int main() {
    std::cout << std::setprecision(8);
    std::cout << "=== Bond Pricing Module Test Suite ===" << std::endl << std::endl;
    
    try {
        testBasicBondPricing();
        testDurationConvexity();
        testAccruedInterest();
        testCarryRoll();
        testBondFutures();
        testINSSBonds();
        testEdgeCases();
        
        std::cout << "==================================" << std::endl;
        std::cout << "All tests passed successfully! ✓" << std::endl;
        std::cout << "BOND_PRICING_OK" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
