# Interpolation Library

A high-performance C++ library providing advanced interpolation methods for mathematical and engineering applications,
with a focus on B-spline interpolation with smoothing capabilities.

## Overview

The interpolation library is part of the CurveForge project and provides robust, production-ready interpolation tools
for multi-dimensional data. It's particularly useful for financial curve construction, computer graphics, trajectory
planning, and scientific data analysis.

## Features

### B-Spline Interpolation (`bspline.h/cpp`)

- **Flexible polynomial degree**: Support for cubic (degree 3) and higher-order B-splines
- **Multiple parameterization strategies**:
    - `uniform`: Evenly spaced parameters (best for uniformly distributed data)
    - `chord`: Chord length parameterization (default, recommended for general use)
    - `centripetal`: Square root of chord lengths (best for non-uniform, sharp corners)
- **Standard interpolation**: Exact fit through all data points
- **Smooth interpolation**: Penalized least-squares fitting with regularization
    - Reduces noise while preserving overall shape
    - Configurable smoothing parameter λ
    - Ideal for noisy financial data or experimental measurements

### Linear Interpolation (`linear.h/cpp`)

- Simple linear interpolation for 1D data
- Fast evaluation for piecewise-linear approximations

## API Reference

### B-Spline Standard Interpolation

Fits a B-spline curve that passes exactly through all given data points.

```cpp
#include "interpolation/bspline.h"
#include <Eigen/Dense>

// Create data points (can be any dimension)
std::vector<Eigen::VectorXd> points;
for (int i = 0; i < 10; ++i) {
    Eigen::Vector2d pt;
    pt << i, i * i;  // Example: parabola
    points.push_back(pt);
}

// Interpolate with cubic B-spline
auto spline = bspline::interpolate(
    points, 
    3,                // degree (3 = cubic)
    "chord"          // parameterization method
);

// Evaluate at any parameter u ∈ [0,1]
Eigen::VectorXd value = spline->evaluate(0.5);
```

**Constructor Parameters:**

- `data_points`: Vector of N-dimensional Eigen vectors (all must have same dimension)
- `degree`: Polynomial degree (typically 3 for cubic splines)
- `parameterization`: One of:
    - `"uniform"`: Parameters uniformly spaced in [0,1]
    - `"chord"`: Parameters based on chord lengths (default, recommended)
    - `"centripetal"`: Parameters based on sqrt of chord lengths

### B-Spline Smooth Interpolation

Fits a smooth B-spline that balances fitting the data points with minimizing curvature.

```cpp
auto smooth_spline = bspline::smooth_interpolate(
    points, 
    3,              // degree
    0.1,            // lambda: smoothing parameter
    "chord"         // parameterization
);

Eigen::VectorXd smoothed_value = smooth_spline->evaluate(0.5);
```

**Smoothing Parameter (λ):**

- `λ = 0.0`: No smoothing (exact interpolation, same as `interpolate()`)
- `λ = 0.01-0.1`: Light smoothing (preserves most features, reduces minor noise)
- `λ = 0.1-1.0`: Moderate smoothing (good for noisy financial/experimental data)
- `λ = 1.0-5.0`: Heavy smoothing (significant noise reduction, may lose detail)
- `λ > 5.0`: Very heavy smoothing (approaches least-squares polynomial fit)

**Algorithm:** Minimizes `||Ax - b||² + λ||Dx||²` where:

- First term: Data fidelity (closeness to original points)
- Second term: Roughness penalty (second derivative = curvature)
- `λ`: Trade-off parameter

### Linear Interpolation

Simple piecewise-linear interpolation for 1D data.

```cpp
#include "interpolation/linear.h"

std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
std::vector<double> y = {0.0, 1.0, 4.0, 9.0};

double interpolated = LinearInterpolation::lininterp(x, y, 1.5);
// Returns 2.5 (linear interpolation between (1,1) and (2,4))
```

## Usage Examples

### Example 1: 2D Curve with Noise Reduction

```cpp
#include "interpolation/bspline.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Generate noisy data (e.g., market observations)
    std::vector<Eigen::VectorXd> noisy_data;
    for (int i = 0; i < 20; ++i) {
        double x = i / 19.0;
        double y = x * x * x;  // True function: cubic
        double noise = 0.05 * (rand() / double(RAND_MAX) - 0.5);
        
        Eigen::Vector2d point;
        point << x, y + noise;
        noisy_data.push_back(point);
    }

    // Exact interpolation (passes through all noisy points)
    auto exact_curve = bspline::interpolate(noisy_data, 3, "chord");

    // Smooth interpolation (reduces noise)
    auto smooth_curve = bspline::smooth_interpolate(
        noisy_data, 
        3,      // cubic
        0.2,    // moderate smoothing
        "chord"
    );

    // Compare results
    std::cout << "At u=0.5:\n";
    std::cout << "  Exact:  " << exact_curve->evaluate(0.5).transpose() << "\n";
    std::cout << "  Smooth: " << smooth_curve->evaluate(0.5).transpose() << "\n";

    return 0;
}
```

### Example 2: 3D Parametric Curve (Helix)

```cpp
#include "interpolation/bspline.h"
#include <Eigen/Dense>
#include <cmath>

// Sample a helix
std::vector<Eigen::VectorXd> helix_points;
for (double t = 0; t < 2*M_PI; t += 0.3) {
    Eigen::Vector3d point;
    point << cos(t), sin(t), t / (2*M_PI);  // Helix in 3D
    helix_points.push_back(point);
}

// Create smooth 3D curve
auto helix_spline = bspline::interpolate(helix_points, 3, "chord");

// Evaluate at 100 points for rendering
for (int i = 0; i <= 100; ++i) {
    double u = i / 100.0;
    Eigen::Vector3d pos = helix_spline->evaluate(u);
    std::cout << pos.transpose() << std::endl;
}
```

### Example 3: Volatility Curve Smoothing (Finance)

```cpp
// Market volatility quotes with noise
std::vector<Eigen::VectorXd> vol_quotes;
std::vector<double> maturities = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
std::vector<double> vols = {0.22, 0.21, 0.20, 0.19, 0.18, 0.17};

for (size_t i = 0; i < maturities.size(); ++i) {
    Eigen::Vector2d point;
    point << maturities[i], vols[i];
    vol_quotes.push_back(point);
}

// Smooth the term structure
auto vol_curve = bspline::smooth_interpolate(vol_quotes, 3, 0.05, "chord");

// Query volatility at any maturity
double vol_at_3y = vol_curve->evaluate(0.6)[1];  // u=0.6 corresponds to ~3Y
std::cout << "3Y volatility: " << (vol_at_3y * 100) << "%\n";
```

## Building

### Requirements

- **C++17** or later
- **Eigen3** library (3.3+)
- **CMake** 3.21+

### CMake Integration

```cmake
# In your CMakeLists.txt
find_package(Eigen3 REQUIRED CONFIG)

add_library(interpolation
        src/bspline.cpp
        src/linear.cpp
)

target_include_directories(interpolation PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(interpolation PUBLIC Eigen3::Eigen)
```

### Building the Library

```bash
cd /path/to/CurveForge
mkdir -p build && cd build
cmake ..
make interpolation -j8
```

## Testing

The library includes comprehensive test suites:

```bash
# Build tests
cd build
make test_bspline test_bspline_smoothing

# Run tests
./tests/interpolation/test_bspline
./tests/interpolation/test_bspline_smoothing
```

### Test Coverage

- ✅ **Standard interpolation accuracy**: Verifies exact pass-through of data points
- ✅ **Smoothing effectiveness**: Validates roughness reduction (15%+ improvement)
- ✅ **Multi-dimensional data**: 2D and 3D curve interpolation
- ✅ **Parameterization methods**: Uniform, chord, and centripetal
- ✅ **Edge cases**: Single point, collinear points, degenerate cases
- ✅ **Numerical stability**: Large datasets, extreme parameter values

### Example Test Output

```
$ ./test_bspline_smoothing
Testing B-spline smoothing with 25 noisy points...
Lambda=0.1: Roughness reduced by 42% ✓
SMOOTH_OK
```

## Mathematical Background

### B-Spline Basis Functions

A B-spline curve of degree `p` is defined as:

**C(u) = Σᵢ Nᵢ,ₚ(u) · Pᵢ**

Where:

- `Nᵢ,ₚ(u)`: B-spline basis functions (computed via Cox-de Boor recursion)
- `Pᵢ`: Control points
- `u ∈ [0,1]`: Parameter

Properties:

- **Local control**: Moving one control point affects only a local region
- **Smoothness**: C^(p-1) continuous (cubic = C² continuous)
- **Convex hull**: Curve lies within convex hull of control points

### Smoothing Formulation

The smooth interpolation solves:

**minimize: ||A·P - D||² + λ·||L·P||²**

Where:

- `A`: Basis function evaluation matrix (m×n)
- `P`: Control points to solve for (n×d, d = dimension)
- `D`: Data points (m×d)
- `L`: Second-difference operator (roughness penalty)
- `λ`: Smoothing parameter (regularization weight)

This is a penalized least-squares problem solved via:

**(AᵀA + λLᵀL)·P = AᵀD**

The solution balances:

1. **Fidelity**: Closeness to original data points
2. **Smoothness**: Minimized second derivative (curvature)

### Visual Comparison: Exact vs Smooth Interpolation

The following figure illustrates the difference between exact interpolation (which passes through all noisy data points)
and smooth interpolation (which reduces noise while preserving the overall trend):

![B-Spline Smoothing Comparison](images/bspline_smoothing_comparison.png)

**Figure 1**: Comparison of smoothed B-spline interpolation vs exact interpolation with noisy data

- **Blue line**: Smoothed interpolation (reduces noise, follows the underlying trend)
- **Orange line**: Exact interpolation with random noise (passes through all noisy points, captures noise)
- **Orange dots**: Noisy data sample points

**Key Observations:**

- The exact interpolation oscillates significantly to pass through every noisy point
- Smooth interpolation filters out noise while maintaining the underlying cubic shape
- Proper choice of smoothing parameter λ balances noise reduction with preserving real features
- The smoothed curve provides a much better representation of the true underlying function
