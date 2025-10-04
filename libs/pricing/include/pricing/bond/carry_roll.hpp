#pragma once
#include "pricing/bond/bond.hpp"

namespace pricing {
    namespace bond {
        /**
         * CarryRollMetrics computes carry and roll analysis for bonds.
         *
         * - Carry: Income earned from holding the bond (coupon income)
         * - Roll: Price change from the passage of time (roll-down the yield curve)
         */
        struct CarryRollMetrics {
            double carry; // Carry (coupon income over the period)
            double roll; // Roll (price change from time passage)
            double total_return; // Total return (carry + roll)
        };

        /**
         * Calculates carry and roll metrics for a bond over a given time horizon, including net carry (carry minus financial cost).
         *
         * @param bond The bond to analyze
         * @param current_yield Current yield to maturity
         * @param forward_yield Expected yield after the time horizon (can be same as current for parallel shift)
         * @param time_horizon Time horizon for the analysis in years (e.g., 0.25 for 3 months)
         * @param funding_rate The annualized funding (financial cost) rate
         * @return CarryRollMetrics structure with carry, roll, and total return
         */
        CarryRollMetrics calculateCarryRoll(const Bond &bond, double current_yield,
                                            double forward_yield, double time_horizon, double funding_rate);

        /**
         * Calculates just the net carry component (coupon income minus financial cost).
         *
         * @param bond The bond
         * @param time_horizon Time horizon in years
         * @param funding_rate The annualized funding (financial cost) rate
         * @return Net carry (coupon income minus financial cost over the period)
         */
        double calculateCarry(const Bond &bond, double time_horizon, double funding_rate);

        /**
         * Calculates just the roll component (price change from time passage).
         *
         * @param bond The bond
         * @param current_yield Current yield to maturity
         * @param forward_yield Expected yield after the time horizon
         * @param time_horizon Time horizon in years
         * @return Roll (price change from time passage)
         */
        double calculateRoll(const Bond &bond, double current_yield,
                             double forward_yield, double time_horizon, double funding_rate = 0.0);
    } // namespace bond
} // namespace pricing
