# TiledCurve Implementation

## Overview

The `TiledCurve` class provides a yield curve representation parameterized by instantaneous forward rates at pillar points. This implementation integrates seamlessly with the CurveForge optimization library, allowing bond and swap pricers to use the calibrated forward rates for discounting.

## Key Features

- **Instantaneous Forward Rate Parameterization**: The curve stores forward rates at pillar points, ensuring smooth and physically meaningful curve construction
- **Accurate Discount Factor Computation**: Uses trapezoidal integration of forward rates to compute discount factors
- **Modular Design**: Works with existing bond and swap pricers through standard discount function interfaces
- **Consistent with Calibration**: Directly uses the forward rates calibrated by the optimizer

## Architecture

### Class: `TiledCurve`

Located in: `libs/pricing/include/pricing/curve/tiled_curve.hpp`

The `TiledCurve` class provides:

```cpp
class TiledCurve {
public:
    // Constructor
    TiledCurve(const std::vector<double>& pillar_times, 
               const std::vector<double>& forwards);
    
    // Core methods
    double discount(double t) const;              // Get discount factor D(t)
    double instantaneous_forward(double t) const; // Get forward rate f(t)
    double get_forward(double t, double dT) const; // Get forward for period [t, t+dT]
    
    // Accessors
    const std::vector<double>& pillarTimes() const;
    const std::vector<double>& forwardRates() const;
};
```

### Integration with Optimizer

The `CalibrationResult` structure has been extended to include:

```cpp
struct CalibrationResult {
    YieldCurve curve;                  // Original discount factor curve
    std::vector<double> pillar_times;  // Pillar times
    std::vector<double> forward_rates; // Calibrated instantaneous forwards
    // ... other fields
};
```

This allows you to create a `TiledCurve` directly from calibration results:

```cpp
auto result = optimizer.calibrate();
TiledCurve tiled_curve(result.pillar_times, result.forward_rates);
```

### Bond Pricing Enhancement

The `Bond` class now includes a new method:

```cpp
double priceFromCurve(const std::function<double(double)>& discount_fn) const;
```

This allows pricing using any discount curve, including `TiledCurve`:

```cpp
Bond bond(100.0, 0.05, 10.0, 2);
auto discount_fn = [&tiled_curve](double t) { return tiled_curve.discount(t); };
double price = bond.priceFromCurve(discount_fn);
```

### Swap Pricing

The `IRSwap` class already supports pricing via discount functions through its `solveDiscount()` method, so it works seamlessly with `TiledCurve`:

```cpp
IRSwap swap(payment_times, 0.04);
auto discount_fn = [&tiled_curve](double t) { return tiled_curve.discount(t); };
double df = swap.solveDiscount(discount_fn);
```

## Mathematical Foundation

### Forward Rate Representation

The curve stores instantaneous forward rates `f(t)` at pillar points `t₁, t₂, ..., tₙ`. Between pillars, forward rates are linearly interpolated:

```
f(t) = f(tᵢ) + (t - tᵢ)/(tᵢ₊₁ - tᵢ) × (f(tᵢ₊₁) - f(tᵢ))    for tᵢ ≤ t ≤ tᵢ₊₁
```

### Discount Factor Computation

Discount factors are computed by integrating the forward rates:

```
D(t) = exp(-∫₀ᵗ f(s) ds)
```

The integral is approximated using the trapezoidal rule:

```
∫₀ᵗ f(s) ds ≈ Σᵢ 0.5 × (f(tᵢ₋₁) + f(tᵢ)) × (tᵢ - tᵢ₋₁)
```

This is the same approach used by the optimizer when building the `YieldCurve`, ensuring consistency between the two representations.

## Usage Example

See `examples/tiled_curve_example.cpp` for a complete demonstration. Here's a minimal example:

```cpp
#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/tiled_curve.hpp"
#include "pricing/bond/bond.hpp"

// 1. Calibrate curve
CurveOptimizer optimizer;
optimizer.add(std::make_shared<OISDeposit>(1.0, 0.03), 0.0, 1.0);
auto result = optimizer.calibrate();

// 2. Create TiledCurve
TiledCurve tiled_curve(result.pillar_times, result.forward_rates);

// 3. Price a bond
Bond bond(100.0, 0.05, 5.0, 2);
auto discount_fn = [&tiled_curve](double t) { return tiled_curve.discount(t); };
double price = bond.priceFromCurve(discount_fn);
```

## Testing

Comprehensive unit tests are provided in `tests/pricing/test_tiled_curve.cpp`:

1. **Basic Construction**: Tests creation and basic discount/forward rate queries
2. **Non-Constant Forwards**: Tests linear interpolation and ordering
3. **Optimizer Integration**: Tests creation from calibration results
4. **Bond Pricing**: Tests bond pricing with tiled curves
5. **Swap Pricing**: Tests swap pricing consistency
6. **Full Workflow**: End-to-end integration test

Run tests with:

```bash
cd build
ctest -R tiled_curve
```

## Performance Considerations

- **Discount Factor Computation**: O(n) where n is the number of pillars before time t
- **Forward Rate Interpolation**: O(log n) using binary search (could be optimized)
- **Memory**: Stores only pillar times and forward rates, minimal overhead

## Comparison with YieldCurve

| Feature | YieldCurve | TiledCurve |
|---------|------------|------------|
| **Representation** | Discount factors at pillars | Instantaneous forwards at pillars |
| **Interpolation** | Log-linear on discount factors | Linear on forward rates |
| **Use Case** | General discount curve | Direct from optimizer calibration |
| **Physical Meaning** | Discount factors | Forward rate structure |
| **Consistency** | Built from forwards | Uses same forwards as calibration |

Both representations are numerically consistent and produce nearly identical results. The choice depends on your use case:

- Use `YieldCurve` for general-purpose discount curves
- Use `TiledCurve` when you need direct access to calibrated forward rates or want to maintain the physical interpretation from the optimization

## Benefits

1. **Modularity**: Integrates seamlessly with existing pricers without modifying their core logic
2. **Consistency**: Uses the same forward rates optimized by the calibration engine
3. **Transparency**: Provides direct access to instantaneous forward rates
4. **Flexibility**: Works with any pricer that accepts a discount function
5. **Physical Interpretation**: Forward rates are more intuitive than discount factors for many applications

## Future Enhancements

Potential areas for future development:

- [ ] Caching of computed discount factors for improved performance
- [ ] Support for different interpolation schemes (cubic spline, flat forward, etc.)
- [ ] Extrapolation beyond the last pillar
- [ ] Parallel curve construction for multi-curve frameworks
- [ ] Analytical gradient computation for sensitivity analysis
