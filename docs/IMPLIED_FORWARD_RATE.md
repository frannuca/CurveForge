# Implied Forward Rate in Bond Futures

## Overview

The **Implied Forward Rate** is a key metric derived from bond futures pricing that represents the interest rate implied by the futures contract for the period from the current date to the futures maturity date.

This feature has been added to the `BondFuture` class to enhance the bond futures analytics capabilities of CurveForge.

## Mathematical Background

### Definition

The implied forward rate is calculated based on the relationship between the spot price of a bond and its forward price as determined by the futures contract:

```
Forward Price = Spot Price × e^(r_forward × T)
```

Where:
- `Forward Price = Futures Price × Conversion Factor`
- `Spot Price` = Current market price of the bond (clean price)
- `r_forward` = Implied forward rate (continuously compounded)
- `T` = Time to futures maturity (in years)

### Formula

Solving for the implied forward rate:

```
r_forward = (1/T) × ln(Forward Price / Spot Price)
r_forward = (1/T) × ln((Futures Price × CF) / Bond Price)
```

### Relationship with Implied Repo Rate

For the cheapest-to-deliver (CTD) bond, the implied forward rate equals the implied repo rate. This is because:

1. The repo rate represents the financing cost to carry the bond to delivery
2. This carry cost is the forward interest rate for the holding period
3. The CTD bond is the one where this relationship is most efficient

For non-CTD bonds, the implied forward rate may differ from the implied repo rate due to delivery optionality and market inefficiencies.

## API Reference

### Function Signature

```cpp
double BondFuture::impliedForwardRate(size_t bond_index, 
                                      double bond_price, 
                                      double futures_price) const;
```

### Parameters

- **bond_index** (`size_t`): Index of the bond in the `deliverable_bonds` vector
  - Must be less than the number of deliverable bonds
  - Throws `std::out_of_range` if invalid

- **bond_price** (`double`): Current market price of the bond (clean price)
  - Must be positive
  - Throws `std::invalid_argument` if non-positive

- **futures_price** (`double`): Current market price of the futures contract
  - Must be positive
  - Throws `std::invalid_argument` if non-positive

### Return Value

Returns the implied forward rate as a `double`, expressed as:
- Annualized rate
- Continuously compounded
- Decimal form (e.g., 0.03 for 3%)

The returned value can be:
- **Positive**: In contango (forward price > spot price)
- **Negative**: In backwardation (forward price < spot price)
- **Zero**: When forward price equals spot price

### Exceptions

| Exception Type | Condition | Message |
|---|---|---|
| `std::out_of_range` | `bond_index >= deliverable_bonds.size()` | "Bond index out of range" |
| `std::invalid_argument` | `bond_price <= 0.0` or `futures_price <= 0.0` | "Prices must be positive" |
| `std::runtime_error` | `futures_maturity_ <= 0.0` | "Futures maturity must be positive" |

## Usage Examples

### Example 1: Basic Usage

```cpp
#include "pricing/bond.hpp"

using namespace pricing::bond;

// Create deliverable bonds
Bond bond1(100.0, 0.06, 10.0, 2);  // 6% coupon, 10-year maturity
Bond bond2(100.0, 0.05, 15.0, 2);  // 5% coupon, 15-year maturity

std::vector<Bond> deliverable_bonds = {bond1, bond2};
std::vector<double> conversion_factors = {1.0, 0.95};

// Create a 6-month bond futures contract
BondFuture future(0.5, deliverable_bonds, conversion_factors);

// Market data
std::vector<double> bond_prices = {105.0, 98.0};
double futures_price = 106.0;

// Calculate implied forward rate for bond 1
double forward_rate = future.impliedForwardRate(0, bond_prices[0], futures_price);

std::cout << "Implied forward rate: " << (forward_rate * 100) << "%" << std::endl;
// Output: Implied forward rate: 1.89%
```

### Example 2: Comparing Multiple Bonds

```cpp
// Calculate forward rates for all deliverable bonds
std::cout << "Implied Forward Rates:\n";
for (size_t i = 0; i < deliverable_bonds.size(); ++i) {
    double fwd_rate = future.impliedForwardRate(i, bond_prices[i], futures_price);
    std::cout << "  Bond " << (i + 1) << ": " 
              << (fwd_rate * 100) << "% p.a.\n";
}
```

### Example 3: Error Handling

```cpp
try {
    // Attempt to calculate with invalid index
    double rate = future.impliedForwardRate(10, 100.0, 105.0);
} catch (const std::out_of_range& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}

try {
    // Attempt to calculate with invalid prices
    double rate = future.impliedForwardRate(0, -100.0, 105.0);
} catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### Example 4: Market Conditions Analysis

```cpp
// Contango (normal market)
double contango_futures = 107.0;
double contango_rate = future.impliedForwardRate(0, 105.0, contango_futures);
std::cout << "Contango forward rate: " << (contango_rate * 100) << "%\n";
// Positive rate expected

// Backwardation (inverted market)
double backwardation_futures = 103.0;
double backwardation_rate = future.impliedForwardRate(0, 105.0, backwardation_futures);
std::cout << "Backwardation forward rate: " << (backwardation_rate * 100) << "%\n";
// Negative rate expected
```

## Practical Applications

### 1. **Yield Curve Construction**
The implied forward rate helps traders and analysts construct forward yield curves from futures prices, which is essential for:
- Interest rate derivatives pricing
- Risk management
- Trading strategy development

### 2. **Arbitrage Detection**
By comparing implied forward rates across different bonds and contracts, traders can identify:
- Mispricing opportunities
- Basis trading opportunities
- Cross-market arbitrage

### 3. **Cheapest-to-Deliver Analysis**
For the CTD bond:
```cpp
size_t ctd_index = future.cheapestToDeliver(bond_prices, repo_rate);
double ctd_forward = future.impliedForwardRate(ctd_index, bond_prices[ctd_index], futures_price);
double ctd_repo = future.impliedRepoRate(ctd_index, bond_prices[ctd_index], futures_price);

// For CTD: ctd_forward ≈ ctd_repo
assert(std::abs(ctd_forward - ctd_repo) < 1e-6);
```

### 4. **Market Sentiment Analysis**
- **Positive forward rates**: Market expects rising rates (contango)
- **Negative forward rates**: Market expects falling rates (backwardation)
- **Magnitude of rate**: Indicates strength of market conviction

## Integration with Existing Metrics

The implied forward rate complements existing bond futures metrics:

| Metric | Purpose | Relationship |
|---|---|---|
| **Implied Repo Rate** | Financing cost to carry bond | Equals forward rate for CTD |
| **Net Basis** | Delivery value differential | Related to forward rate spread |
| **Futures Price** | Contract value | Input to forward rate calculation |
| **Cheapest to Deliver** | Optimal delivery choice | Bond with lowest implied forward cost |

## Technical Notes

### Continuous Compounding

The implementation uses continuous compounding for mathematical convenience and consistency with standard derivatives pricing:

```
e^(r × T) growth factor
```

To convert to simple or discrete compounding:

```cpp
// Continuous to simple rate
double simple_rate = (std::exp(continuous_rate * T) - 1.0) / T;

// Continuous to discrete (annual compounding)
double discrete_rate = std::exp(continuous_rate) - 1.0;
```

### Negative Rates

The function correctly handles negative forward rates, which can occur in:
- Backwardated markets
- Inverted yield curves
- Low or negative interest rate environments

### Precision

The function uses double precision floating-point arithmetic. For very small time periods or rates:
- Relative error: ~1e-15
- Absolute error: ~1e-12

## Testing

Comprehensive tests are included in `tests/pricing/test_bond_pricing.cpp`:

1. **Basic calculation tests**: Verify formula correctness
2. **Edge case tests**: Zero rates, negative rates, extreme values
3. **Error handling tests**: Invalid inputs, out-of-range indices
4. **Consistency tests**: Relationship with repo rate for CTD
5. **Market condition tests**: Contango and backwardation scenarios

Run tests with:
```bash
./build/tests/run_bond_pricing_tests
```

## See Also

- `BondFuture::impliedRepoRate()` - Calculate implied repo rate
- `BondFuture::futuresPrice()` - Calculate theoretical futures price
- `BondFuture::cheapestToDeliver()` - Identify CTD bond
- `BondFuture::netBasis()` - Calculate net basis

## References

1. Hull, J. (2018). *Options, Futures, and Other Derivatives*. Pearson.
2. Fabozzi, F. J. (2012). *Bond Markets, Analysis, and Strategies*. Pearson.
3. Tuckman, B., & Serrat, A. (2011). *Fixed Income Securities*. Wiley.
