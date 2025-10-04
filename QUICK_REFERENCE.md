# Bond Pricing Quick Reference

## Include

```cpp
#include "pricing/bond.hpp"
using namespace pricing::bond;
```

## 1. Basic Bond Pricing

### Create a Bond

```cpp
// Bond(face_value, coupon_rate, maturity, payment_frequency)
Bond bond(100.0, 0.05, 10.0, 2);  // 5% coupon, 10y, semi-annual
```

### Price from Yield

```cpp
double price = bond.priceFromYield(0.04);  // Price at 4% yield
```

### Yield from Price

```cpp
double yield = bond.yieldFromPrice(108.0);  // Yield at $108 price
```

### Duration & Convexity

```cpp
double duration = bond.duration(0.05);
double mod_duration = bond.modifiedDuration(0.05);
double convexity = bond.convexity(0.05);
```

### Accrued Interest

```cpp
double accrued = bond.accruedInterest(0.25);  // At t=0.25 years
```

## 2. Carry and Roll Analysis

### Calculate Carry and Roll (Net Carry)

```cpp
// calculateCarryRoll(bond, current_yield, forward_yield, time_horizon, funding_rate)
// Carry is net of financial cost: coupon income minus funding cost over the horizon
CarryRollMetrics metrics = calculateCarryRoll(bond, 0.05, 0.045, 0.5, 0.03); // 3% funding rate

std::cout << "Carry (net): " << metrics.carry << std::endl;
std::cout << "Roll: " << metrics.roll << std::endl;
std::cout << "Total Return: " << metrics.total_return << std::endl;
```

### Separate Calculations

```cpp
double carry = calculateCarry(bond, 0.5);  // 6-month carry
double roll = calculateRoll(bond, 0.05, 0.045, 0.5);
```

## 3. Bond Futures

### Create Deliverable Bonds

```cpp
Bond bond1(100.0, 0.06, 10.0, 2);
Bond bond2(100.0, 0.05, 15.0, 2);
std::vector<Bond> deliverable_bonds = {bond1, bond2};
```

### Calculate Conversion Factors

```cpp
std::vector<double> conversion_factors;
for (const auto& b : deliverable_bonds) {
    conversion_factors.push_back(calculateConversionFactor(b, 0.06));
}
```

### Create Futures Contract

```cpp
// BondFuture(futures_maturity, deliverable_bonds, conversion_factors)
BondFuture future(0.5, deliverable_bonds, conversion_factors);
```

### Futures Pricing

```cpp
std::vector<double> bond_prices = {105.0, 98.0};
double futures_price = future.futuresPrice(bond_prices, 0.03);  // 3% repo
```

### Cheapest to Deliver

```cpp
size_t ctd = future.cheapestToDeliver(bond_prices, 0.03);
std::cout << "CTD: Bond " << (ctd + 1) << std::endl;
```

### Implied Repo Rate

```cpp
double implied_repo = future.impliedRepoRate(0, bond_prices[0], futures_price);
```

### Net Basis

```cpp
double net_basis = future.netBasis(0, bond_prices[0], futures_price, 0.5);
```

## 4. INSS Bonds (Brazilian Social Security)

### Create INSS Bond

```cpp
// INSSBond(face, coupon, maturity, freq, tax_rate, is_floating)
INSSBond inss(100.0, 0.05, 10.0, 2, 0.15, false);  // 15% tax
```

### INSS Pricing

```cpp
double price = inss.priceFromYield(0.05);  // After-tax price
double yield = inss.yieldFromPrice(price); // After-tax yield
```

### After-Tax Coupon

```cpp
double after_tax = inss.afterTaxCoupon(2.5);  // 2.5 * (1 - 0.15)
```

### INSS Metrics

```cpp
INSSMetrics metrics = calculateINSSMetrics(inss, price);
std::cout << "Net yield: " << metrics.net_yield << std::endl;
std::cout << "Gross yield: " << metrics.gross_yield << std::endl;
std::cout << "Tax PV: " << metrics.tax_pv << std::endl;
```

## Error Handling

All functions throw exceptions on errors:

```cpp
try {
    Bond bond(-100.0, 0.05, 10.0, 2);  // Invalid face value
} catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

Common exceptions:

- `std::invalid_argument` - Invalid parameters
- `std::runtime_error` - Numerical issues
- `std::out_of_range` - Index out of bounds

## Build & Test

### Build

```bash
cmake -S . -B build
cmake --build build --parallel
```

### Run Tests

```bash
ctest --test-dir build -R bond
# Or directly:
./build/tests/run_bond_pricing_tests
```

### Run Examples

```bash
./build/examples/bond_pricing_examples
```

## Documentation

- **Full API Docs**: `libs/pricing/BOND_PRICING_README.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Code Examples**: `examples/bond_pricing_examples.cpp`
- **Tests**: `tests/pricing/test_bond_pricing.cpp`
