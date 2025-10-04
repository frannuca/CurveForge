# Bond Pricing Module

## Overview

The Bond Pricing Module provides comprehensive functionality for pricing and analyzing fixed-income securities. It includes support for:

1. **Basic Bond Pricing** - Yield-to-price and price-to-yield conversions
2. **Carry and Roll Analysis** - Income and price change metrics
3. **Bond Futures Pricing** - Futures contracts, implied repo rates, and cheapest-to-deliver
4. **INSS Bond Pricing** - Brazilian social security bonds with tax considerations

## Features

### 1. Basic Bond Pricing (`bond.hpp`)

The `Bond` class represents a fixed-coupon bond with periodic payments.

#### Key Features:
- Yield to price conversion using discrete compounding
- Price to yield conversion using Newton-Raphson method
- Accrued interest calculation
- Duration (Macaulay and Modified)
- Convexity calculation

#### Example Usage:

```cpp
#include "pricing/bond.hpp"

using namespace pricing::bond;

// Create a 5% coupon bond, $100 face value, 10 years to maturity, semi-annual payments
Bond bond(100.0, 0.05, 10.0, 2);

// Calculate price at 4% yield
double price = bond.priceFromYield(0.04);
std::cout << "Price at 4% yield: $" << price << std::endl;
// Output: Price at 4% yield: $108.18

// Calculate yield from price
double yield = bond.yieldFromPrice(108.0);
std::cout << "Yield at $108 price: " << (yield * 100) << "%" << std::endl;
// Output: Yield at $108 price: 4.03%

// Calculate duration and convexity
double duration = bond.duration(0.05);
double mod_duration = bond.modifiedDuration(0.05);
double convexity = bond.convexity(0.05);

std::cout << "Duration: " << duration << " years" << std::endl;
std::cout << "Modified Duration: " << mod_duration << std::endl;
std::cout << "Convexity: " << convexity << std::endl;
```

### 2. Carry and Roll Analysis (`carry_roll.hpp`)

Analyze the expected return from holding a bond over a specific time horizon.

#### Components:
- **Carry**: Income from coupon payments
- **Roll**: Price change from the passage of time (rolling down the yield curve)
- **Total Return**: Carry + Roll

#### Example Usage:

```cpp
#include "pricing/bond/carry_roll.hpp"

using namespace pricing::bond;

Bond bond(100.0, 0.05, 10.0, 2);

// Calculate carry and roll over 6 months
// Assuming current yield is 5% and forward yield remains at 5%
CarryRollMetrics metrics = calculateCarryRoll(bond, 0.05, 0.05, 0.5);

std::cout << "Carry: $" << metrics.carry << std::endl;
std::cout << "Roll: $" << metrics.roll << std::endl;
std::cout << "Total Return: $" << metrics.total_return << std::endl;

// Output:
// Carry: $2.50 (one coupon payment)
// Roll: $0.00 (minimal price change when yield is unchanged)
// Total Return: $2.50

// Calculate with yield curve change
CarryRollMetrics metrics2 = calculateCarryRoll(bond, 0.05, 0.045, 0.5);
std::cout << "Roll with yield decline: $" << metrics2.roll << std::endl;
```

### 3. Bond Futures Pricing (`bond_future.hpp`)

Price bond futures contracts and identify the cheapest-to-deliver bond.

#### Key Features:
- Futures pricing based on cost-of-carry
- Implied repo rate calculation
- Cheapest-to-deliver bond identification
- Net basis calculation
- Conversion factor calculation

#### Example Usage:

```cpp
#include "pricing/bond/bond_future.hpp"

using namespace pricing::bond;

// Create deliverable bonds
Bond bond1(100.0, 0.06, 10.0, 2);
Bond bond2(100.0, 0.05, 15.0, 2);

std::vector<Bond> deliverable_bonds = {bond1, bond2};

// Calculate conversion factors (at 6% notional yield)
std::vector<double> conversion_factors;
for (const auto& bond : deliverable_bonds) {
    conversion_factors.push_back(calculateConversionFactor(bond, 0.06));
}

// Create futures contract (6 months to expiry)
BondFuture future(0.5, deliverable_bonds, conversion_factors);

// Current bond prices
std::vector<double> bond_prices = {105.0, 98.0};

// Calculate theoretical futures price (assuming 3% repo rate)
double futures_price = future.futuresPrice(bond_prices, 0.03);
std::cout << "Futures price: $" << futures_price << std::endl;

// Identify cheapest to deliver
size_t ctd_index = future.cheapestToDeliver(bond_prices, 0.03);
std::cout << "Cheapest to deliver: Bond " << (ctd_index + 1) << std::endl;

// Calculate implied repo rate for a specific bond
double implied_repo = future.impliedRepoRate(0, bond_prices[0], futures_price);
std::cout << "Implied repo rate: " << (implied_repo * 100) << "%" << std::endl;

// Calculate net basis
double net_basis = future.netBasis(0, bond_prices[0], futures_price, 0.5);
std::cout << "Net basis: $" << net_basis << std::endl;
```

### 4. INSS Bond Pricing (`inss.hpp`)

Price INSS (Instituto Nacional do Seguro Social) bonds, which are Brazilian social security bonds with specific tax treatment.

#### Key Features:
- Tax-adjusted pricing (taxes on coupon payments)
- After-tax yield calculation
- Tax present value calculation
- Comprehensive INSS metrics

#### Example Usage:

```cpp
#include "pricing/bond/inss.hpp"

using namespace pricing::bond;

// Create INSS bond with 15% tax rate
// 5% coupon, $100 face, 10 years, semi-annual payments, 15% tax
INSSBond inss_bond(100.0, 0.05, 10.0, 2, 0.15, false);

// Calculate price at 5% yield (after-tax)
double price = inss_bond.priceFromYield(0.05);
std::cout << "INSS Bond price: $" << price << std::endl;

// Calculate comprehensive metrics
INSSMetrics metrics = calculateINSSMetrics(inss_bond, price);

std::cout << "Net yield: " << (metrics.net_yield * 100) << "%" << std::endl;
std::cout << "Gross yield: " << (metrics.gross_yield * 100) << "%" << std::endl;
std::cout << "Tax PV: $" << metrics.tax_pv << std::endl;
std::cout << "Duration: " << metrics.duration << " years" << std::endl;
std::cout << "Convexity: " << metrics.convexity << std::endl;

// Calculate after-tax coupon
double after_tax = inss_bond.afterTaxCoupon(2.5);
std::cout << "After-tax coupon: $" << after_tax << std::endl;
// Output: After-tax coupon: $2.125 (2.5 * 0.85)
```

## Technical Details

### Discounting Convention

The module uses **discrete compounding** matching the bond's payment frequency:

- Discount factor: `DF(t) = (1 + y/f)^(-f*t)`
- Where `y` is the yield, `f` is the payment frequency, and `t` is the time in years

This ensures that a bond priced at its coupon rate yields exactly par value.

### Duration Formulas

- **Macaulay Duration**: `D = Σ(t_i * PV_i) / Price`
- **Modified Duration**: `D_mod = D / (1 + y/f)`

Where `PV_i` is the present value of cash flow at time `t_i`.

### Convexity Formula

`C = Σ(t_i^2 * PV_i) / (Price * (1 + y/f)^2)`

## Error Handling

All functions include comprehensive error checking:
- Invalid parameters throw `std::invalid_argument`
- Numerical issues throw `std::runtime_error`
- Out-of-range access throws `std::out_of_range`

## Building

The bond pricing module is included in the main `pricing` library:

```cmake
target_link_libraries(your_target
    PRIVATE
    CurveForge::pricing
)
```

## Testing

Comprehensive tests are provided in `tests/pricing/test_bond_pricing.cpp`:

```bash
# Run all tests
ctest --test-dir build

# Run bond pricing tests specifically
./build/tests/run_bond_pricing_tests
```

## Future Extensions

The module is designed to be extensible. Potential future additions:

1. **Callable/Putable Bonds** - Options embedded in bonds
2. **Floating Rate Notes** - Bonds with variable coupon rates
3. **Inflation-Linked Bonds** - TIPS and similar securities
4. **Credit Risk** - Default probability and recovery rates
5. **Convertible Bonds** - Bonds with equity conversion features

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
- Fabozzi, F. J. (2012). *Bond Markets, Analysis, and Strategies*
- Brazilian Central Bank regulations for INSS securities
