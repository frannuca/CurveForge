# Quick Start Guide: CurveForge Optimization Library

## Installation

### Prerequisites
```bash
sudo apt-get install libnlopt-dev libeigen3-dev
```

### Build
```bash
cd CurveForge
mkdir build && cd build
cmake .. -DCURVEFORGE_BUILD_APPS=OFF
make -j4
```

### Run Tests
```bash
ctest -R optimizer
```

### Run Examples
```bash
./examples/optimizer_example
```

## 5-Minute Tutorial

### Step 1: Include Headers
```cpp
#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include "pricing/curve/fra.hpp"
#include "pricing/curve/irswap.hpp"
```

### Step 2: Create Optimizer
```cpp
using namespace pricing;

CurveOptimizer optimizer;
```

### Step 3: Add Market Instruments
```cpp
// Short-end: Overnight Index Swap deposits
optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);  // 3M @ 2.5%
optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);   // 6M @ 2.8%
optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);   // 1Y @ 3.0%

// Mid-section: Forward Rate Agreements
optimizer.add(std::make_shared<FRA>(0.5, 1.5, 0.032), 0.0, 1.0);     // 6Mx18M @ 3.2%

// Long-end: Interest Rate Swaps
std::vector<double> swap5y = {1.0, 2.0, 3.0, 4.0, 5.0};
optimizer.add(std::make_shared<IRSwap>(swap5y, 0.035), 0.0, 1.0);    // 5Y @ 3.5%
```

### Step 4: Calibrate
```cpp
auto result = optimizer.calibrate();
```

### Step 5: Use Results
```cpp
if (result.success) {
    // Access calibrated curve
    double df_2y = result.curve.discount(2.0);
    
    // Check fit quality
    std::cout << "Objective value: " << result.objective_value << std::endl;
    
    // Examine residuals
    for (size_t i = 0; i < result.residuals.size(); ++i) {
        std::cout << "Instrument " << i << " residual: " 
                  << result.residuals[i] << std::endl;
    }
} else {
    std::cerr << "Calibration failed: " << result.message << std::endl;
}
```

## Advanced: Custom Configuration

```cpp
CurveOptimizer::Config config;
config.max_iterations = 2000;           // More iterations
config.relative_tolerance = 1e-7;       // Tighter tolerance
config.initial_forward_rate = 0.04;     // Better initial guess

CurveOptimizer optimizer(config);
// ... add instruments ...
auto result = optimizer.calibrate();
```

## Advanced: Weighted Calibration

```cpp
// Emphasize accuracy for short-end instruments
optimizer.add(deposit_3m, 0.0, 10.0);   // 10x weight
optimizer.add(deposit_6m, 0.0, 10.0);   // 10x weight
optimizer.add(swap_10y, 0.0, 1.0);      // Normal weight
```

## What the Library Does

1. **Extracts Pillar Times**: From instrument maturities
2. **Parameterizes Curve**: Using instantaneous forward rates at pillars
3. **Optimizes**: Minimizes weighted least-squares error
4. **Returns**: Calibrated YieldCurve + diagnostics

## Key Features

✅ **Automatic pillar detection** from instrument maturities  
✅ **Smooth curves** via instantaneous forward interpolation  
✅ **Robust optimization** using NLopt COBYLA algorithm  
✅ **Flexible weighting** to emphasize critical instruments  
✅ **Comprehensive diagnostics** (residuals, iterations, objective)  
✅ **Multiple instrument types** (deposits, FRAs, swaps)  

## Typical Calibration Quality

- **Objective values**: < 1e-6 for well-specified problems
- **Convergence**: 10-100 iterations
- **Speed**: < 0.01 seconds for typical curves
- **Residuals**: Individual errors typically < 1e-4

## Common Patterns

### Pattern 1: Short-End Only
```cpp
CurveOptimizer optimizer;
optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);
auto result = optimizer.calibrate();
```

### Pattern 2: Full Term Structure
```cpp
CurveOptimizer optimizer;
// Deposits for 0-1Y
// FRAs for 1-2Y
// Swaps for 2Y+
auto result = optimizer.calibrate();
```

### Pattern 3: High Precision Short-End
```cpp
CurveOptimizer optimizer;
optimizer.add(deposit_overnight, 0.0, 100.0);  // Very high weight
optimizer.add(deposit_1w, 0.0, 50.0);          // High weight
optimizer.add(swap_10y, 0.0, 1.0);             // Normal weight
auto result = optimizer.calibrate();
```

## Troubleshooting

### Low Convergence Quality
- Increase `max_iterations`
- Adjust `initial_forward_rate` closer to expected rates
- Check instrument consistency

### Unrealistic Rates
- Verify market data inputs
- Check instrument specifications
- Review residuals to identify problematic instruments

### Optimization Failure
- Check that instruments cover the desired time range
- Ensure no duplicate maturities
- Verify all rates are positive

## Next Steps

1. Read the full documentation: `docs/OPTIMIZER_README.md`
2. Run the examples: `./examples/optimizer_example`
3. Review the tests: `tests/pricing/test_optimizer.cpp`
4. Check the implementation: `libs/pricing/src/optimizer.cpp`

## API Summary

| Class | Purpose |
|-------|---------|
| `CurveOptimizer` | Main optimization engine |
| `CalibrationResult` | Result with curve + diagnostics |
| `Config` | Optimization parameters |
| `CalibrationInstrument` | Instrument + market data wrapper |

| Method | Returns | Description |
|--------|---------|-------------|
| `add(instrument, price, weight)` | `CurveOptimizer&` | Add instrument to calibration |
| `calibrate()` | `CalibrationResult` | Perform optimization |
| `pillarTimes()` | `vector<double>` | Get pillar times |

| Result Fields | Type | Description |
|--------------|------|-------------|
| `curve` | `YieldCurve` | Calibrated curve |
| `success` | `bool` | Convergence flag |
| `message` | `string` | Status message |
| `objective_value` | `double` | Final objective |
| `residuals` | `vector<double>` | Per-instrument errors |

## Support

For questions or issues:
- Check documentation in `docs/OPTIMIZER_README.md`
- Review examples in `examples/optimizer_example.cpp`
- Examine tests in `tests/pricing/test_optimizer.cpp`
