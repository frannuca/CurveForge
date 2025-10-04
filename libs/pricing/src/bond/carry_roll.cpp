#include "pricing/bond/carry_roll.hpp"
#include <cmath>
#include <stdexcept>

namespace pricing {
    namespace bond {
        CarryRollMetrics calculateCarryRoll(const Bond &bond, double current_yield,
                                            double forward_yield, double time_horizon, double funding_rate) {
            if (time_horizon <= 0.0) {
                throw std::invalid_argument("Time horizon must be positive");
            }
            if (time_horizon > bond.maturity()) {
                throw std::invalid_argument("Time horizon exceeds bond maturity");
            }

            CarryRollMetrics metrics;

            // Calculate net carry (coupon income minus financial cost)
            metrics.carry = calculateCarry(bond, time_horizon, funding_rate);

            // Calculate roll (price change)
            metrics.roll = calculateRoll(bond, current_yield, forward_yield, time_horizon);

            // Total return
            metrics.total_return = metrics.carry + metrics.roll;

            return metrics;
        }

        double calculateCarry(const Bond &bond, double time_horizon, double funding_rate) {
            if (time_horizon <= 0.0) {
                throw std::invalid_argument("Time horizon must be positive");
            }

            double gross_carry = 0.0;
            // Sum up all coupon payments that occur within the time horizon
            for (size_t i = 0; i < bond.couponTimes().size(); ++i) {
                double t = bond.couponTimes()[i];
                if (t <= time_horizon) {
                    double coupon = bond.couponAmounts()[i];
                    // Don't include principal repayment in carry
                    if (std::abs(t - bond.maturity()) < 1e-10) {
                        coupon -= bond.faceValue();
                    }
                    gross_carry += coupon;
                }
            }
            // Financial cost: cost of funding the bond position over the time horizon
            double price = bond.priceFromYield(funding_rate);
            double financial_cost = price * funding_rate * time_horizon;
            double net_carry = gross_carry - financial_cost;
            return net_carry;
        }

        double calculateRoll(const Bond &bond, double current_yield,
                             double forward_yield, double time_horizon, double funding_rate) {
            if (time_horizon <= 0.0) {
                throw std::invalid_argument("Time horizon must be positive");
            }
            if (time_horizon > bond.maturity()) {
                throw std::invalid_argument("Time horizon exceeds bond maturity");
            }

            // Current price
            double current_price = bond.priceFromYield(current_yield);

            // Create a new bond with reduced maturity (after time_horizon has passed)
            // This represents the bond at the future date
            double new_maturity = bond.maturity() - time_horizon;

            if (new_maturity <= 1e-10) {
                // Bond matures within the horizon, roll includes principal
                double carry = calculateCarry(bond, time_horizon, funding_rate);
                double total_cf = 0.0;
                for (size_t i = 0; i < bond.couponTimes().size(); ++i) {
                    if (bond.couponTimes()[i] <= time_horizon + 1e-10) {
                        total_cf += bond.couponAmounts()[i];
                    }
                }
                return total_cf - carry - current_price;
            }

            // Calculate future price with reduced maturity at forward yield
            // Shift all cash flows back by time_horizon
            // Find the payment frequency from the first period
            double period = bond.couponTimes()[0];
            int frequency = std::round(1.0 / period);

            double future_price = 0.0;
            for (size_t i = 0; i < bond.couponTimes().size(); ++i) {
                double t = bond.couponTimes()[i];
                if (t > time_horizon) {
                    double cf = bond.couponAmounts()[i];
                    double future_t = t - time_horizon;
                    double df = std::pow(1.0 + forward_yield / frequency, -frequency * future_t);
                    future_price += cf * df;
                }
            }

            // Roll is the price change (excluding coupon income)
            double roll = future_price - current_price;

            return roll;
        }
    } // namespace bond
} // namespace pricing
