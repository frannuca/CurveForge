# CurveForge Documentation

Welcome to the CurveForge documentation. This quantitative finance library provides comprehensive tools for curve
construction, volatility surface calibration, and financial instrument pricing.

## Library Documentation

### Core Libraries

- **[Interpolation Library](INTERPOLATION.md)** - B-spline and linear interpolation with smoothing capabilities
    - Standard and smooth B-spline interpolation
    - Multiple parameterization strategies
    - Applications in curve construction and data smoothing

- **[Volatility Library](VOLATILITY.md)** - Implied volatility surface calibration
    - Black-Scholes option pricing
    - Implied volatility calculation (Newton-Raphson and Brent's method)
    - Volatility surface interpolation and calibration
    - Multiple surface parametrizations

### Additional Resources

- **Project README**: See root `README.txt` for build instructions and project overview
- **Implementation Summary**: See `VOLATILITY_IMPLEMENTATION_SUMMARY.md` in project root

## Quick Start

### Building the Libraries

```bash
cd /path/to/CurveForge
mkdir -p build && cd build
cmake ..
make -j8
```

### Running Examples

```bash
# Volatility surface calibration
./examples/vol_surface_example

# B-spline smoothing demonstration
./examples/bspline_smoothing_demo
```

### Running Tests

```bash
cd build
make test
ctest
```

## Key Features

### Interpolation

- 🎯 Exact B-spline interpolation through data points
- 📊 Smoothing with configurable regularization parameter λ
- 🔄 Multi-dimensional support (2D, 3D, N-D curves)
- ⚡ Efficient evaluation using Cox-de Boor recursion

### Volatility Surface Calibration

- 📈 Black-Scholes option pricing (European calls/puts)
- 🎲 Robust implied volatility calculation
- 🗺️ Surface interpolation (bilinear, bicubic spline, variance)
- 💼 Multiple parametrizations (strike, moneyness, log-moneyness)

## Applications

### Financial Engineering

- Yield curve construction and smoothing
- Volatility smile and term structure modeling
- Forward curve interpolation
- Option pricing and risk management

### Scientific Computing

- Experimental data smoothing
- Noisy signal filtering
- Trend extraction from time series

### Computer Graphics

- Smooth path generation
- Animation curve design
- Font rendering and vector graphics

## Project Structure

```
CurveForge/
├── docs/                          # Documentation (you are here)
│   ├── INTERPOLATION.md          # Interpolation library guide
│   ├── VOLATILITY.md             # Volatility library guide
│   └── images/                   # Documentation figures
├── libs/
│   ├── interpolation/            # B-spline interpolation
│   ├── volatility/               # Vol surface calibration
│   ├── curve/                    # Curve construction
│   ├── pricing/                  # Pricing engines
│   ├── instruments/              # Financial instruments
│   └── time/                     # Date/time utilities
├── examples/                     # Example programs
├── tests/                        # Unit tests
└── build/                        # Build artifacts
```

## Dependencies

- **C++17** or later
- **CMake** 3.21+
- **Eigen3** 3.3+ (linear algebra)
- **Xerces-C** (XML parsing for data contracts)
- **Boost** (utilities)

## Contributing

Contributions are welcome! Please ensure:

- All tests pass (`make test && ctest`)
- Code follows existing style conventions
- New features include tests and documentation

## License

See LICENSE file in project root.

## Author

Francisco Nunez  
CurveForge - Quantitative Finance Library  
2025

