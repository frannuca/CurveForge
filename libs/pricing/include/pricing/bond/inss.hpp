#pragma once
#include "pricing/bond/bond.hpp"

namespace pricing {
namespace bond {

/**
 * INSSBond represents an INSS (Instituto Nacional do Seguro Social) bond.
 * These are Brazilian social security bonds with specific characteristics.
 * 
 * INSS bonds typically have:
 * - Fixed or floating rate coupons
 * - Specific payment schedules
 * - Tax treatment considerations
 */
class INSSBond {
    Bond underlying_bond_;    // Underlying bond structure
    double tax_rate_;        // Tax rate applicable to coupon payments (e.g., 0.15 for 15%)
    bool is_floating_rate_;  // Whether the bond has a floating rate

public:
    /**
     * Constructs an INSS bond with specified parameters.
     * 
     * @param face_value Face/par value of the bond
     * @param coupon_rate Annual coupon rate (for fixed-rate bonds)
     * @param maturity Time to maturity in years
     * @param payment_frequency Number of coupon payments per year
     * @param tax_rate Tax rate on coupon payments (e.g., 0.15 for 15%)
     * @param is_floating_rate Whether this is a floating rate bond
     */
    INSSBond(double face_value, double coupon_rate, double maturity, 
             int payment_frequency, double tax_rate, bool is_floating_rate = false);

    /**
     * Calculates the INSS bond price given a yield, accounting for taxes.
     * 
     * @param yield Yield to maturity
     * @return Bond price (after-tax)
     */
    double priceFromYield(double yield) const;

    /**
     * Calculates the yield from price for INSS bonds, accounting for taxes.
     * 
     * @param price Bond price (after-tax)
     * @param initial_guess Initial guess for yield
     * @return Yield to maturity
     */
    double yieldFromPrice(double price, double initial_guess = -1.0) const;

    /**
     * Calculates the after-tax coupon payment amount.
     * 
     * @param coupon_payment Gross coupon payment
     * @return After-tax coupon payment
     */
    double afterTaxCoupon(double coupon_payment) const;

    /**
     * Calculates the effective yield accounting for tax effects.
     * 
     * @param gross_yield Gross yield
     * @return Effective after-tax yield
     */
    double effectiveYield(double gross_yield) const;

    /**
     * Calculates the present value of tax payments.
     * 
     * @param yield Yield to maturity
     * @return Present value of taxes
     */
    double taxPV(double yield) const;

    // Accessors
    const Bond& underlyingBond() const { return underlying_bond_; }
    double taxRate() const { return tax_rate_; }
    bool isFloatingRate() const { return is_floating_rate_; }
};

/**
 * Metrics specific to INSS bond analysis.
 */
struct INSSMetrics {
    double gross_yield;      // Gross yield (before tax)
    double net_yield;        // Net yield (after tax)
    double tax_pv;          // Present value of tax payments
    double duration;        // Duration
    double convexity;       // Convexity
};

/**
 * Calculates comprehensive metrics for an INSS bond.
 * 
 * @param bond The INSS bond
 * @param price Current market price
 * @return INSSMetrics structure with all relevant metrics
 */
INSSMetrics calculateINSSMetrics(const INSSBond& bond, double price);

} // namespace bond
} // namespace pricing
