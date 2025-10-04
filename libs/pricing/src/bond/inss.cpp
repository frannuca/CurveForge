#include "pricing/bond/inss.hpp"
#include <cmath>
#include <stdexcept>

namespace pricing {
namespace bond {

INSSBond::INSSBond(double face_value, double coupon_rate, double maturity, 
                   int payment_frequency, double tax_rate, bool is_floating_rate)
    : underlying_bond_(face_value, coupon_rate, maturity, payment_frequency),
      tax_rate_(tax_rate),
      is_floating_rate_(is_floating_rate) {
    
    if (tax_rate < 0.0 || tax_rate > 1.0) {
        throw std::invalid_argument("Tax rate must be between 0 and 1");
    }
}

double INSSBond::priceFromYield(double yield) const {
    // For INSS bonds, we need to account for taxes on coupon payments
    // Price = PV of after-tax coupons + PV of principal
    
    const auto& coupon_times = underlying_bond_.couponTimes();
    const auto& coupon_amounts = underlying_bond_.couponAmounts();
    
    // Find the payment frequency from the first period
    double period = coupon_times[0];
    int frequency = std::round(1.0 / period);
    
    double price = 0.0;
    double face_value = underlying_bond_.faceValue();
    
    for (size_t i = 0; i < coupon_times.size(); ++i) {
        double t = coupon_times[i];
        double cf = coupon_amounts[i];
        double df = std::pow(1.0 + yield / frequency, -frequency * t);
        
        // Check if this is the final payment (includes principal)
        if (std::abs(t - underlying_bond_.maturity()) < 1e-10) {
            // Last payment: after-tax coupon + full principal
            double coupon = cf - face_value;
            double after_tax_coupon = afterTaxCoupon(coupon);
            double total_cf = after_tax_coupon + face_value;
            price += total_cf * df;
        } else {
            // Regular coupon payment with tax
            double after_tax_cf = afterTaxCoupon(cf);
            price += after_tax_cf * df;
        }
    }
    
    return price;
}

double INSSBond::yieldFromPrice(double price, double initial_guess) const {
    if (price <= 0.0) {
        throw std::invalid_argument("Price must be positive");
    }

    // Use coupon rate as initial guess if not provided
    double y = (initial_guess < 0.0) ? underlying_bond_.couponRate() : initial_guess;
    
    const auto& coupon_times = underlying_bond_.couponTimes();
    const auto& coupon_amounts = underlying_bond_.couponAmounts();
    double face_value = underlying_bond_.faceValue();
    
    // Find the payment frequency from the first period
    double period = coupon_times[0];
    int frequency = std::round(1.0 / period);
    
    const int max_iterations = 100;
    const double tolerance = 1e-8;
    
    // Newton-Raphson method
    for (int iter = 0; iter < max_iterations; ++iter) {
        double p = 0.0;      // Price
        double dp_dy = 0.0;  // Derivative of price w.r.t. yield
        
        for (size_t i = 0; i < coupon_times.size(); ++i) {
            double t = coupon_times[i];
            double cf = coupon_amounts[i];
            
            double after_tax_cf;
            if (std::abs(t - underlying_bond_.maturity()) < 1e-10) {
                double coupon = cf - face_value;
                after_tax_cf = afterTaxCoupon(coupon) + face_value;
            } else {
                after_tax_cf = afterTaxCoupon(cf);
            }
            
            double n = frequency * t;
            double base = 1.0 + y / frequency;
            double df = std::pow(base, -n);
            
            p += after_tax_cf * df;
            dp_dy -= after_tax_cf * t * std::pow(base, -n - 1);
        }
        
        double f = p - price;
        
        if (std::abs(f) < tolerance) {
            return y;
        }
        
        if (std::abs(dp_dy) < 1e-14) {
            throw std::runtime_error("Derivative too small in Newton-Raphson");
        }
        
        y = y - f / dp_dy;
        
        // Keep yield reasonable
        if (y < -0.5 || y > 2.0) {
            throw std::runtime_error("Yield iteration diverged");
        }
    }
    
    throw std::runtime_error("Yield calculation did not converge");
}

double INSSBond::afterTaxCoupon(double coupon_payment) const {
    // After-tax coupon = Coupon * (1 - tax_rate)
    return coupon_payment * (1.0 - tax_rate_);
}

double INSSBond::effectiveYield(double gross_yield) const {
    // For a simplified model, effective yield can be approximated
    // This is a simplified version; actual calculation would depend on specific tax treatment
    return gross_yield * (1.0 - tax_rate_);
}

double INSSBond::taxPV(double yield) const {
    // Calculate the present value of all tax payments
    const auto& coupon_times = underlying_bond_.couponTimes();
    const auto& coupon_amounts = underlying_bond_.couponAmounts();
    double face_value = underlying_bond_.faceValue();
    
    // Find the payment frequency from the first period
    double period = coupon_times[0];
    int frequency = std::round(1.0 / period);
    
    double tax_pv = 0.0;
    
    for (size_t i = 0; i < coupon_times.size(); ++i) {
        double t = coupon_times[i];
        double cf = coupon_amounts[i];
        
        // Tax is only on coupon, not principal
        double coupon = cf;
        if (std::abs(t - underlying_bond_.maturity()) < 1e-10) {
            coupon = cf - face_value;
        }
        
        double tax = coupon * tax_rate_;
        double df = std::pow(1.0 + yield / frequency, -frequency * t);
        tax_pv += tax * df;
    }
    
    return tax_pv;
}

INSSMetrics calculateINSSMetrics(const INSSBond& bond, double price) {
    INSSMetrics metrics;
    
    // Calculate yields
    metrics.net_yield = bond.yieldFromPrice(price);
    
    // Gross yield (approximation based on underlying bond)
    // This is simplified; more accurate would require iterative calculation
    metrics.gross_yield = metrics.net_yield / (1.0 - bond.taxRate());
    
    // Tax PV
    metrics.tax_pv = bond.taxPV(metrics.net_yield);
    
    // Duration and convexity (using underlying bond as approximation)
    metrics.duration = bond.underlyingBond().duration(metrics.net_yield);
    metrics.convexity = bond.underlyingBond().convexity(metrics.net_yield);
    
    return metrics;
}

} // namespace bond
} // namespace pricing
