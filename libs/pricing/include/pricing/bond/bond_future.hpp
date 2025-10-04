#pragma once
#include "pricing/bond/bond.hpp"
#include <vector>

namespace pricing {
namespace bond {

/**
 * BondFuture represents a bond futures contract.
 * 
 * Provides functionality to:
 * - Price bond futures
 * - Calculate conversion factors
 * - Compute implied repo rates
 * - Identify cheapest-to-deliver bond
 */
class BondFuture {
    double futures_maturity_;      // Time to futures expiry in years
    std::vector<Bond> deliverable_bonds_;  // Bonds eligible for delivery
    std::vector<double> conversion_factors_; // Conversion factors for each bond

public:
    /**
     * Constructs a BondFuture contract.
     * 
     * @param futures_maturity Time to futures expiry in years
     * @param deliverable_bonds Vector of bonds that can be delivered
     * @param conversion_factors Conversion factors for each deliverable bond
     */
    BondFuture(double futures_maturity, 
               const std::vector<Bond>& deliverable_bonds,
               const std::vector<double>& conversion_factors);

    /**
     * Calculates the theoretical futures price using the cheapest-to-deliver bond.
     * 
     * @param bond_prices Current market prices of deliverable bonds
     * @param repo_rate Financing/repo rate for carrying the bond
     * @return Theoretical futures price
     */
    double futuresPrice(const std::vector<double>& bond_prices, double repo_rate) const;

    /**
     * Calculates the implied repo rate given the futures price and a specific bond.
     * 
     * @param bond_index Index of the bond in deliverable_bonds
     * @param bond_price Current market price of the bond
     * @param futures_price Current futures price
     * @return Implied repo rate
     */
    double impliedRepoRate(size_t bond_index, double bond_price, double futures_price) const;

    /**
     * Identifies the cheapest-to-deliver bond index.
     * 
     * @param bond_prices Current market prices of deliverable bonds
     * @param repo_rate Financing/repo rate
     * @return Index of the cheapest-to-deliver bond
     */
    size_t cheapestToDeliver(const std::vector<double>& bond_prices, double repo_rate) const;

    /**
     * Calculates the net basis for a specific bond.
     * Net Basis = Bond Price - (Futures Price × Conversion Factor) - Accrued Interest
     * 
     * @param bond_index Index of the bond
     * @param bond_price Current market price of the bond
     * @param futures_price Current futures price
     * @param accrued_interest Accrued interest on the bond
     * @return Net basis
     */
    double netBasis(size_t bond_index, double bond_price, 
                    double futures_price, double accrued_interest) const;

    // Accessors
    double futuresMaturity() const { return futures_maturity_; }
    const std::vector<Bond>& deliverableBonds() const { return deliverable_bonds_; }
    const std::vector<double>& conversionFactors() const { return conversion_factors_; }
};

/**
 * Calculates the conversion factor for a bond in a futures contract.
 * The conversion factor normalizes bonds with different coupons and maturities.
 * 
 * @param bond The bond
 * @param notional_coupon The notional coupon rate of the futures contract (e.g., 0.06 for 6%)
 * @return Conversion factor
 */
double calculateConversionFactor(const Bond& bond, double notional_coupon);

} // namespace bond
} // namespace pricing
