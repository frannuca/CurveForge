# CurveForge Optimization Library

## Overview

The CurveForge Optimization Library provides a sophisticated framework for calibrating yield curves to market instrument prices using Sequential Quadratic Programming (SQP). The library minimizes the least square error between observed market prices and model-computed prices while ensuring the curve is smooth and well-behaved through regularization techniques.

## Key Features

- **SQP Algorithm**: Uses NLopt's SLSQP (Sequential Least Squares Programming), a gradient-based SQP method for efficient optimization
- **Curve Regularization**: Supports first-order and second-order regularization for smooth forward curves
- **Instantaneous Forward Calibration**: Parameterizes curves using instantaneous forward rates for smoothness
- **Modular Design**: Clean separation between optimization logic and instrument pricing
- **Flexible Configuration**: Customizable convergence criteria, regularization strength, and optimization parameters
- **Comprehensive Diagnostics**: Detailed residuals and convergence information
- **Multi-Instrument Support**: Works with deposits, FRAs, swaps, and other curve instruments

## Architecture

### Core Components

1. **CurveOptimizer**: Main optimization engine using SQP algorithm
2. **CalibrationInstrument**: Wrapper combining instruments with market data
3. **CalibrationResult**: Contains calibrated curve and diagnostic information
4. **Config**: Configuration for optimization parameters, including regularization settings

### Design Principles

- **SQP Optimization**: Uses Sequential Quadratic Programming for fast convergence with gradient information
- **Instantaneous Forward Parameterization**: The curve is represented using instantaneous forward rates at pillar points (extracted from instrument maturities)
- **Discount Factor Conversion**: Forward rates are converted to discount factors using:
  ```
  D(t) = exp(-∫₀ᵗ f(s) ds)
  ```
  where the integral is approximated using the trapezoidal rule
- **Regularized Objective**: Minimizes weighted sum of squared errors plus regularization penalty:
  ```
  Objective = Σᵢ wᵢ(DFₘₐᵣₖₑₜ - DFcurve)² + λ × Regularization
  ```
  where the regularization term penalizes non-smooth forward curves

## Usage

### Basic Example: Single Deposit

```cpp
#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"

using namespace pricing;

// Create optimizer
CurveOptimizer optimizer;

// Add a 1-year OIS deposit at 3%
auto deposit = std::make_shared<OISDeposit>(1.0, 0.03);
optimizer.add(deposit, 0.0, 1.0);  // instrument, market_price (unused), weight

// Calibrate
auto result = optimizer.calibrate();

if (result.success) {
    std::cout << "Calibration succeeded!" << std::endl;
    std::cout << "DF at 1Y: " << result.curve.discount(1.0) << std::endl;
}
```

### Advanced Example: Full Curve

```cpp
#include "pricing/curve/optimizer.hpp"
#include "pricing/curve/ois_deposit.hpp"
#include "pricing/curve/fra.hpp"
#include "pricing/curve/irswap.hpp"

using namespace pricing;

// Configure optimizer
CurveOptimizer::Config config;
config.max_iterations = 1000;
config.relative_tolerance = 1e-6;
config.initial_forward_rate = 0.03;

CurveOptimizer optimizer(config);

// Short-end: deposits
optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 1.0);
optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 1.0);
optimizer.add(std::make_shared<OISDeposit>(1.0, 0.030), 0.0, 1.0);

// Mid-section: FRAs
optimizer.add(std::make_shared<FRA>(0.5, 1.5, 0.032), 0.0, 1.0);
optimizer.add(std::make_shared<FRA>(1.0, 2.0, 0.033), 0.0, 1.0);

// Long-end: swaps
std::vector<double> swap5y = {1.0, 2.0, 3.0, 4.0, 5.0};
optimizer.add(std::make_shared<IRSwap>(swap5y, 0.035), 0.0, 1.0);

// Calibrate
auto result = optimizer.calibrate();

// Examine results
std::cout << "Success: " << result.success << std::endl;
std::cout << "Message: " << result.message << std::endl;
std::cout << "Objective: " << result.objective_value << std::endl;

for (size_t i = 0; i < result.residuals.size(); ++i) {
    std::cout << "Residual " << i << ": " << result.residuals[i] << std::endl;
}
```

### Weighted Calibration

You can assign different weights to instruments to emphasize fit quality:

```cpp
// High weight for short-end accuracy
optimizer.add(std::make_shared<OISDeposit>(0.25, 0.025), 0.0, 10.0);
optimizer.add(std::make_shared<OISDeposit>(0.5, 0.028), 0.0, 10.0);

// Normal weight for long-end
optimizer.add(std::make_shared<OISDeposit>(5.0, 0.040), 0.0, 1.0);
```

### Regularization for Smooth Curves

Control curve smoothness using regularization parameters:

```cpp
CurveOptimizer::Config config;
config.regularization_lambda = 0.01;     // Regularization strength
config.regularization_order = 2;         // Second-order (smooth curvature)

CurveOptimizer optimizer(config);

// Add instruments...
auto result = optimizer.calibrate();

// The resulting curve will be smoother due to regularization
```

**Regularization Options:**
- `regularization_lambda = 0.0`: No regularization (pure data fitting)
- `regularization_lambda > 0.0`: Increasing values produce smoother curves
- `regularization_order = 1`: Penalize changes in forward rates (first derivative)
- `regularization_order = 2`: Penalize changes in slope (second derivative, default)

## API Reference

### CurveOptimizer

#### Constructor
```cpp
explicit CurveOptimizer(const Config& config = Config())
```
Creates an optimizer with specified configuration.

#### Methods

**add**
```cpp
CurveOptimizer& add(const CurveInstrumentPtr& instrument, 
                    double market_price, 
                    double weight = 1.0)
```
Adds a calibration instrument with market price and weight.

**calibrate**
```cpp
CalibrationResult calibrate()
```
Performs SQP optimization and returns calibrated curve with diagnostics.

**pillarTimes**
```cpp
const std::vector<double>& pillarTimes() const
```
Returns the pillar times used for parameterization.

### Config

Configuration options for the optimizer:

- `relative_tolerance` (double): Relative convergence tolerance (default: 1e-6)
- `absolute_tolerance` (double): Absolute convergence tolerance (default: 1e-8)
- `max_iterations` (int): Maximum optimization iterations (default: 1000)
- `initial_forward_rate` (double): Initial guess for forward rates (default: 0.03)
- `regularization_lambda` (double): Regularization strength for curve smoothing (default: 0.01)
- `regularization_order` (int): Order of regularization - 1 (first derivative) or 2 (second derivative, default)

### CalibrationResult

Result structure containing:

- `curve` (YieldCurve): Calibrated yield curve
- `objective_value` (double): Final objective function value
- `residuals` (vector<double>): Individual residuals for each instrument
- `iterations` (int): Number of optimization iterations performed
- `success` (bool): Whether optimization converged
- `message` (string): Status message

### CalibrationInstrument

Structure holding:

- `instrument` (CurveInstrumentPtr): The curve instrument
- `market_price` (double): Observed market price
- `weight` (double): Weight in objective function

## Implementation Details

### Optimization Algorithm

The library uses **SLSQP** (Sequential Least Squares Programming) from NLopt:
- Gradient-based SQP (Sequential Quadratic Programming) method
- Uses gradient information computed via finite differences
- Handles bound constraints on forward rates (0.1% to 20%)
- Efficient and robust for smooth curve calibration problems

### Forward Rate Parameterization

Instantaneous forward rates are defined at pillar times (instrument maturities). Between pillars, linear interpolation is used in the integral approximation:

```
D(tᵢ) = D(tᵢ₋₁) × exp(-0.5 × (f(tᵢ₋₁) + f(tᵢ)) × (tᵢ - tᵢ₋₁))
```

This ensures:
- Smooth forward curves
- Positive discount factors
- No-arbitrage conditions

### Objective Function

The objective minimizes the sum of data fitting error and regularization penalty:

```
Objective = Σᵢ wᵢ × (DFₘₐᵣₖₑₜ(Tᵢ) - DFcurve(Tᵢ))² + λ × R(f)
```

where:
- `wᵢ` is the instrument weight
- `DFₘₐᵣₖₑₜ(Tᵢ)` is the discount factor implied by market instrument
- `DFcurve(Tᵢ)` is the discount factor from calibrated curve
- `λ` is the regularization parameter
- `R(f)` is the regularization term

### Regularization Terms

**First-order regularization** (smoothness in forward rates):
```
R₁(f) = Σᵢ (f[i] - f[i-1])²
```

**Second-order regularization** (smoothness in curvature):
```
R₂(f) = Σᵢ (f[i+1] - 2f[i] + f[i-1])²
```

Second-order regularization is recommended for most applications as it produces curves with smooth curvature while allowing gradual changes in forward rates.

## Performance Considerations

- **Pillar Selection**: Number of pillars equals number of unique instrument maturities
- **Convergence**: Typical calibration converges in 10-50 iterations with SQP
- **Objective Values**: Well-calibrated curves achieve objectives < 1e-6
- **Gradient Computation**: Uses finite differences with epsilon = 1e-7

## Testing

The library includes comprehensive unit tests covering:
- Single instrument calibration
- Multi-instrument curves
- FRA and swap instruments
- Residual computation
- Configuration options

Run tests with:
```bash
cd build
ctest -R optimizer
```

## Dependencies

- **NLopt**: Nonlinear optimization library (version 2.7+)
- **C++20**: Modern C++ standard
- **CurveForge::pricing**: Core pricing library

## Building

The optimization library is automatically built with the pricing library when NLopt is available:

```cmake
find_package(NLopt REQUIRED)
target_link_libraries(pricing PUBLIC NLopt::nlopt)
```

## Future Enhancements

Potential extensions:
- Additional optimization algorithms (L-BFGS, SLSQP)
- Gradient-based optimization for faster convergence
- Regularization terms for smoother curves
- Multi-curve calibration (OIS, LIBOR, etc.)
- Parallel optimization for bootstrap initialization

## References

- NLopt Documentation: https://nlopt.readthedocs.io/
- Curve Bootstrapping: Andersen & Piterbarg, "Interest Rate Modeling"
- Optimization Methods: Nocedal & Wright, "Numerical Optimization"

## License

This library is part of the CurveForge project and follows the same license terms.
