#pragma once

/**
 * Bond Pricing Module
 * 
 * This module provides comprehensive bond pricing and analysis functionality:
 * 
 * 1. Basic Bond Pricing (bond.hpp):
 *    - Yield to price conversion
 *    - Price to yield conversion
 *    - Duration and convexity calculations
 *    - Accrued interest
 * 
 * 2. Carry and Roll Analysis (carry_roll.hpp):
 *    - Carry metrics (coupon income)
 *    - Roll metrics (price change over time)
 *    - Total return analysis
 * 
 * 3. Bond Futures (bond_future.hpp):
 *    - Futures pricing
 *    - Implied repo rate calculation
 *    - Cheapest-to-deliver identification
 *    - Conversion factor calculation
 *    - Net basis calculation
 * 
 * 4. INSS Bonds (inss.hpp):
 *    - Brazilian social security bond pricing
 *    - Tax-adjusted pricing and yields
 *    - INSS-specific metrics
 * 
 * Example Usage:
 * 
 * ```cpp
 * #include "pricing/bond.hpp"
 * 
 * // Create a 5% coupon bond maturing in 10 years
 * pricing::bond::Bond bond(100.0, 0.05, 10.0, 2);
 * 
 * // Calculate price from yield
 * double price = bond.priceFromYield(0.04);
 * 
 * // Calculate carry and roll
 * auto metrics = pricing::bond::calculateCarryRoll(bond, 0.04, 0.045, 0.25);
 * 
 * // Work with bond futures
 * std::vector<pricing::bond::Bond> deliverable_bonds = { bond };
 * std::vector<double> conversion_factors = { 1.05 };
 * pricing::bond::BondFuture future(0.5, deliverable_bonds, conversion_factors);
 * ```
 */

#include "pricing/bond/bond.hpp"
#include "pricing/bond/carry_roll.hpp"
#include "pricing/bond/bond_future.hpp"
#include "pricing/bond/inss.hpp"
