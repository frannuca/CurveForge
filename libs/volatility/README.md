# Volatility Library - Implied Volatility Surface Calibration

## Overview

This library provides a comprehensive implementation of implied volatility surface calibration based on option market
prices. It includes Black-Scholes option pricing, implied volatility calculation using multiple numerical methods, and
flexible surface interpolation capabilities.

## Features

### 1. Black-Scholes Option Pricing (`BlackScholes.h/cpp`)

- **Option Pricing**: European call and put option pricing
- **Greeks Calculation**: Delta, gamma, vega calculations
- **Implied Volatility**: Two robust methods for computing implied volatility:
    - **Newton-Raphson Method**: Fast convergence for well-behaved cases
    - **Brent's Method**: More robust for edge cases with guaranteed bracketing

### 2. Implied Volatility Surface (`ImpliedVolSurface.h/cpp`)

#### Surface Parametrizations

The library supports three different parametrization spaces:

- **STRIKE_SPACE**: Direct strike and maturity (K, T)
- **MONEYNESS_SPACE**: Relative moneyness K/F and maturity
- **LOG_MONEYNESS_SPACE**: Log-moneyness ln(K/F) and maturity (recommended)

#### Interpolation Methods

- **BILINEAR**: Fast bilinear interpolation on a 2D grid
- **BICUBIC_SPLINE**: Smooth cubic spline interpolation using B-splines
- **LINEAR_IN_VARIANCE**: Interpolate in variance space (σ²·T) for no-arbitrage

#### Key Capabilities

1. **Calibration**: Extract implied volatilities from market option prices
2. **Interpolation**: Query volatility for any strike/maturity combination
3. **Export/Import**: Serialize surfaces to/from XSD-based data contracts
4. **Statistics**: Compute calibration quality metrics (mean error, RMSE, max error)
5. **Validation**: Check for arbitrage-free conditions (TODO)

## Usage Examples

### Basic Calibration

```cpp
#include "volatility/ImpliedVolSurface.h"
#include "volatility/BlackScholes.h"

using namespace curve::volatility;

// Create option market quotes
std::vector<OptionQuote> quotes;
OptionQuote quote;
quote.strike = 100.0;
quote.maturity = 1.0;
quote.market_price = 10.45;
quote.spot = 100.0;
quote.forward = 105.0;
quote.is_call = true;
quotes.push_back(quote);

// Create and calibrate surface
ImpliedVolSurface surface(
    ImpliedVolSurface::SurfaceType::LOG_MONEYNESS_SPACE,
    ImpliedVolSurface::InterpolationMethod::BICUBIC_SPLINE,
    0.05  // risk-free rate
);

if (surface.calibrate(quotes)) {
    std::cout << "Calibration successful!" << std::endl;
    
    // Query volatility at any point
    double vol = surface.get_volatility(105.0, 1.5, 105.0);
    std::cout << "Volatility: " << (vol * 100) << "%" << std::endl;
}
```

### Computing Implied Volatility

```cpp
#include "volatility/BlackScholes.h"

using namespace curve::volatility;

double market_price = 10.45;
double spot = 100.0;
double strike = 105.0;
double risk_free_rate = 0.05;
double maturity = 1.0;

// Method 1: Newton-Raphson (fast)
double implied_vol = BlackScholes::implied_volatility(
    market_price, spot, strike, risk_free_rate, maturity, true
);

// Method 2: Brent's method (robust)
double implied_vol_robust = BlackScholes::implied_volatility_brent(
    market_price, spot, strike, risk_free_rate, maturity, true
);
```

### Export to Market Data Format

```cpp
// Export calibrated surface to XSD format
xml_schema::date as_of(2025, 11, 29);
auto vol_surface = surface.export_to_vol_surface("EURUSD", as_of);

// The surface can now be serialized to XML or used in market data snapshots
```

## Implementation Details

### Calibration Process

1. **Extract Implied Volatilities**: For each market quote, compute the implied volatility that matches the market price
   using Black-Scholes
2. **Build Grid**: Organize calibrated points into a 2D grid structure
3. **Construct Interpolator**: Build splines or prepare bilinear interpolation weights
4. **Validate**: Check for arbitrage opportunities (butterfly spreads, calendar spreads)

### Interpolation Strategy

For **BICUBIC_SPLINE** method:

- Build cubic B-splines along the strike dimension for each maturity
- Linearly interpolate between maturity slices
- This provides smooth surfaces while maintaining computational efficiency

For **BILINEAR** method:

- Standard bilinear interpolation on the volatility grid
- Fast but less smooth than splines

For **LINEAR_IN_VARIANCE** method:

- Interpolate total variance (σ²·T) linearly
- Convert back to volatility
- Helps prevent calendar arbitrage

### Numerical Stability

- Automatic fallback from Newton-Raphson to Brent's method when vega is too small
- Clamping of volatility guesses to reasonable bounds (0.01% to 1000%)
- Graceful handling of edge cases (zero time to maturity, deep ITM/OTM)

## Data Structures

### OptionQuote

```cpp
struct OptionQuote {
    double strike;          // Strike price
    double maturity;        // Time to maturity (years)
    double market_price;    // Market option price
    double spot;            // Current spot price
    double forward;         // Forward price
    bool is_call;           // Call (true) or Put (false)
    double moneyness;       // Computed moneyness
};
```

### VolPoint

```cpp
struct VolPoint {
    double strike;          // Strike price
    double maturity;        // Time to maturity
    double volatility;      // Implied volatility
    double moneyness;       // Moneyness coordinate
};
```

### CalibrationStats

```cpp
struct CalibrationStats {
    double mean_error;      // Mean pricing error
    double max_error;       // Maximum pricing error
    double rmse;            // Root mean square error
    int num_points;         // Number of calibrated points
};
```

## Dependencies

- **Eigen3**: Matrix operations and linear algebra
- **datacontracts**: XSD-based data serialization (vol.xsd, marketdata.xsd)
- **interpolation**: B-spline interpolation library
- Standard C++17 features

## Building

```bash
cd build
cmake ..
make volatility -j8
```

## Testing

Run the comprehensive example:

```bash
cd build/examples
./vol_surface_example
```

This demonstrates:

1. Basic surface calibration with volatility smile
2. Interpolation at intermediate points
3. Comparison of interpolation methods
4. Direct implied volatility calculation

## Future Enhancements

1. **Arbitrage Validation**: Implement butterfly and calendar arbitrage checks
2. **Advanced Models**: Support for stochastic volatility models (Heston, SABR)
3. **Local Volatility**: Convert implied vol surface to local volatility
4. **Greeks from Surface**: Surface-level delta, gamma, vega calculations
5. **American Options**: Pricing and implied volatility for American-style options
6. **Dividend Handling**: Support for discrete dividends in forward calculations
7. **Multi-Asset**: Correlation surfaces for basket options
8. **Smile Dynamics**: Term structure of volatility smiles

## References

1. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
2. Rebonato, R. (2004). "Volatility and Correlation"
3. Hagan, P. et al. (2002). "Managing Smile Risk" - SABR model
4. Dupire, B. (1994). "Pricing with a Smile"

## License

See project-level LICENSE file.

## Author

Francisco Nunez - November 29, 2025

