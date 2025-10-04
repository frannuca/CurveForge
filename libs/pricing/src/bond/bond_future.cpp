#include "pricing/bond/bond_future.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace pricing {
namespace bond {

BondFuture::BondFuture(double futures_maturity, 
                       const std::vector<Bond>& deliverable_bonds,
                       const std::vector<double>& conversion_factors)
    : futures_maturity_(futures_maturity),
      deliverable_bonds_(deliverable_bonds),
      conversion_factors_(conversion_factors) {
    
    if (futures_maturity <= 0.0) {
        throw std::invalid_argument("Futures maturity must be positive");
    }
    if (deliverable_bonds.empty()) {
        throw std::invalid_argument("Must have at least one deliverable bond");
    }
    if (deliverable_bonds.size() != conversion_factors.size()) {
        throw std::invalid_argument("Number of bonds and conversion factors must match");
    }
    
    for (double cf : conversion_factors) {
        if (cf <= 0.0) {
            throw std::invalid_argument("Conversion factors must be positive");
        }
    }
}

double BondFuture::futuresPrice(const std::vector<double>& bond_prices, double repo_rate) const {
    if (bond_prices.size() != deliverable_bonds_.size()) {
        throw std::invalid_argument("Number of prices must match number of deliverable bonds");
    }

    // Find the cheapest to deliver
    size_t ctd_index = cheapestToDeliver(bond_prices, repo_rate);
    
    // Calculate futures price based on CTD
    double bond_price = bond_prices[ctd_index];
    double cf = conversion_factors_[ctd_index];
    
    // Forward price = Spot price * exp(repo_rate * time) / CF
    // Simplified: futures price based on cost of carry
    double forward_price = bond_price * std::exp(repo_rate * futures_maturity_);
    double futures_price = forward_price / cf;
    
    return futures_price;
}

double BondFuture::impliedRepoRate(size_t bond_index, double bond_price, 
                                   double futures_price) const {
    if (bond_index >= deliverable_bonds_.size()) {
        throw std::out_of_range("Bond index out of range");
    }
    if (bond_price <= 0.0 || futures_price <= 0.0) {
        throw std::invalid_argument("Prices must be positive");
    }

    double cf = conversion_factors_[bond_index];
    
    // Forward price = Futures price * CF
    double forward_price = futures_price * cf;
    
    // Solve for repo rate: forward_price = bond_price * exp(repo * T)
    // => repo = ln(forward_price / bond_price) / T
    if (forward_price <= 0.0 || bond_price <= 0.0) {
        throw std::runtime_error("Invalid prices for repo calculation");
    }
    
    double repo = std::log(forward_price / bond_price) / futures_maturity_;
    
    return repo;
}

double BondFuture::impliedForwardRate(size_t bond_index, double bond_price, 
                                      double futures_price) const {
    if (bond_index >= deliverable_bonds_.size()) {
        throw std::out_of_range("Bond index out of range");
    }
    if (bond_price <= 0.0 || futures_price <= 0.0) {
        throw std::invalid_argument("Prices must be positive");
    }
    if (futures_maturity_ <= 0.0) {
        throw std::runtime_error("Futures maturity must be positive");
    }

    double cf = conversion_factors_[bond_index];
    
    // Forward price = Futures price * CF
    double forward_price = futures_price * cf;
    
    if (forward_price <= bond_price) {
        // If forward price <= spot price, the implied forward rate is non-positive
        // This is valid in some market conditions (e.g., inverted yield curves)
        // We still calculate it but it will be <= 0
    }
    
    // Calculate implied forward rate using continuous compounding
    // forward_price = bond_price * exp(r_forward * T)
    // => r_forward = ln(forward_price / bond_price) / T
    double forward_rate = std::log(forward_price / bond_price) / futures_maturity_;
    
    return forward_rate;
}

size_t BondFuture::cheapestToDeliver(const std::vector<double>& bond_prices, 
                                     double repo_rate) const {
    if (bond_prices.size() != deliverable_bonds_.size()) {
        throw std::invalid_argument("Number of prices must match number of deliverable bonds");
    }

    double min_cost = std::numeric_limits<double>::max();
    size_t ctd_index = 0;
    
    for (size_t i = 0; i < deliverable_bonds_.size(); ++i) {
        // Cost to deliver = Forward bond price - (Futures price * CF)
        // We want to minimize this, which is equivalent to maximizing implied repo
        // Simplification: use forward price / CF as proxy for cost
        double forward_price = bond_prices[i] * std::exp(repo_rate * futures_maturity_);
        double normalized_price = forward_price / conversion_factors_[i];
        
        if (normalized_price < min_cost) {
            min_cost = normalized_price;
            ctd_index = i;
        }
    }
    
    return ctd_index;
}

double BondFuture::netBasis(size_t bond_index, double bond_price, 
                           double futures_price, double accrued_interest) const {
    if (bond_index >= deliverable_bonds_.size()) {
        throw std::out_of_range("Bond index out of range");
    }

    double cf = conversion_factors_[bond_index];
    
    // Net Basis = Bond Price - (Futures Price × CF) - Accrued Interest
    // Actually: Net Basis = (Bond Price + AI) - (Futures Price × CF)
    // But often calculated as: Invoice Price - (Futures × CF)
    double net_basis = bond_price - (futures_price * cf) - accrued_interest;
    
    return net_basis;
}

double calculateConversionFactor(const Bond& bond, double notional_coupon) {
    if (notional_coupon <= 0.0) {
        throw std::invalid_argument("Notional coupon must be positive");
    }

    // Conversion factor is the price of the bond at the notional coupon yield
    // rounded to a specified number of decimal places (typically 4)
    double cf = bond.priceFromYield(notional_coupon) / bond.faceValue();
    
    // Round to 4 decimal places
    cf = std::round(cf * 10000.0) / 10000.0;
    
    return cf;
}

} // namespace bond
} // namespace pricing
