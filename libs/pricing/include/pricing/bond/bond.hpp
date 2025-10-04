#pragma once
#include <vector>
#include <functional>

namespace pricing {
namespace bond {

/**
 * Bond represents a fixed-coupon bond with periodic coupon payments.
 * 
 * This class provides functionality to:
 * - Calculate bond price from yield (yield to price conversion)
 * - Calculate bond yield from price (price to yield conversion)
 * - Compute accrued interest
 * - Calculate duration and convexity
 */
class Bond {
    double face_value_;           // Face/par value of the bond
    double coupon_rate_;          // Annual coupon rate (e.g., 0.05 for 5%)
    double maturity_;             // Time to maturity in years
    std::vector<double> coupon_times_;  // Payment times in years
    std::vector<double> coupon_amounts_; // Coupon amounts

public:
    /**
     * Constructs a Bond with specified parameters.
     * 
     * @param face_value Face/par value of the bond
     * @param coupon_rate Annual coupon rate (e.g., 0.05 for 5%)
     * @param maturity Time to maturity in years
     * @param payment_frequency Number of coupon payments per year (e.g., 2 for semi-annual)
     */
    Bond(double face_value, double coupon_rate, double maturity, int payment_frequency = 2);

    /**
     * Calculates the bond price given a yield to maturity.
     * 
     * @param yield Yield to maturity (annual, e.g., 0.04 for 4%)
     * @return Bond price (clean price, without accrued interest)
     */
    double priceFromYield(double yield) const;

    /**
     * Calculates the yield to maturity given a bond price.
     * Uses Newton-Raphson method for root finding.
     * 
     * @param price Bond price (clean price)
     * @param initial_guess Initial guess for yield (default: coupon rate)
     * @return Yield to maturity
     */
    double yieldFromPrice(double price, double initial_guess = -1.0) const;

    /**
     * Calculates accrued interest at a given time.
     * 
     * @param t Current time (years from issue)
     * @return Accrued interest amount
     */
    double accruedInterest(double t) const;

    /**
     * Calculates the Macaulay duration of the bond.
     * 
     * @param yield Yield to maturity
     * @return Macaulay duration in years
     */
    double duration(double yield) const;

    /**
     * Calculates the modified duration of the bond.
     * 
     * @param yield Yield to maturity
     * @return Modified duration
     */
    double modifiedDuration(double yield) const;

    /**
     * Calculates the convexity of the bond.
     * 
     * @param yield Yield to maturity
     * @return Convexity
     */
    double convexity(double yield) const;

    // Accessors
    double faceValue() const { return face_value_; }
    double couponRate() const { return coupon_rate_; }
    double maturity() const { return maturity_; }
    const std::vector<double>& couponTimes() const { return coupon_times_; }
    const std::vector<double>& couponAmounts() const { return coupon_amounts_; }
};

} // namespace bond
} // namespace pricing
