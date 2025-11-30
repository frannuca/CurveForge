# B-Spline Mathematical Formulas

This document describes the mathematical formulas implemented in `bspline.cpp`.

## Table of Contents

1. [B-Spline Curve Definition](#b-spline-curve-definition)
2. [Basis Functions (Cox-de Boor Recursion)](#basis-functions-cox-de-boor-recursion)
3. [Knot Vector Construction](#knot-vector-construction)
4. [Standard Interpolation](#standard-interpolation)
5. [Smooth Interpolation (Penalized Fitting)](#smooth-interpolation-penalized-fitting)
6. [Parameterization Methods](#parameterization-methods)

---

## B-Spline Curve Definition

A B-spline curve **C(u)** of degree **p** is defined as:

```
C(u) = Σ(i=0 to n) Nᵢ,ₚ(u) · Pᵢ
```

Where:

- **u** ∈ [0,1] is the parameter
- **Nᵢ,ₚ(u)** are the B-spline basis functions of degree p
- **Pᵢ** are the control points (vectors in ℝᵈ)
- **n+1** is the number of control points

### Implementation

```cpp
// In evaluate() method (lines 70-87)
Eigen::VectorXd bspline::evaluate(double u) const {
    // De Boor's algorithm evaluates C(u) efficiently
    // Working with p+1 relevant control points in local support
    for (int r = 1; r <= p; ++r) {
        for (int j = p; j >= r; --j) {
            double alpha = (u - U[k - p + j]) / (U[k + 1 + j - r] - U[k - p + j]);
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }
    return d[p];
}
```

---

## Basis Functions (Cox-de Boor Recursion)

B-spline basis functions are computed recursively:

### Base Case (degree 0):

```
Nᵢ,₀(u) = { 1  if uᵢ ≤ u < uᵢ₊₁
          { 0  otherwise
```

### Recursive Case (degree p):

```
Nᵢ,ₚ(u) = [(u - uᵢ)/(uᵢ₊ₚ - uᵢ)] · Nᵢ,ₚ₋₁(u) + [(uᵢ₊ₚ₊₁ - u)/(uᵢ₊ₚ₊₁ - uᵢ₊₁)] · Nᵢ₊₁,ₚ₋₁(u)
```

Where:

- **uᵢ** are the knot values
- Division by zero is handled as 0/0 = 0

### Implementation

```cpp
// In basis_function() method (lines 106-124)
std::vector<double> bspline::basis_function(double u) const {
    std::vector<double> N(p + 1, 0.0);
    std::vector<double> left(p + 1), right(p + 1);
    N[0] = 1.0;
    
    for (int j = 1; j <= p; ++j) {
        left[j] = u - U[k + 1 - j];
        right[j] = U[k + j] - u;
        double saved = 0.0;
        
        for (int r = 0; r < j; ++r) {
            double den = right[r + 1] + left[j - r];
            double temp = (den == 0.0) ? 0.0 : N[r] / den;
            N[r] = saved + temp * right[r + 1];
            saved = temp * left[j - r];
        }
        N[j] = saved;
    }
    return N;
}
```

---

## Knot Vector Construction

### Clamped Uniform Knot Vector

For **n+1** control points and degree **p**, the knot vector **U** has size **m+1 = n+p+2**:

```
U = [0, 0, ..., 0,  u_{p+1}, u_{p+2}, ..., u_{n},  1, 1, ..., 1]
     └─ p+1 zeros ─┘                                └─ p+1 ones ─┘
```

Interior knots are uniformly spaced:

```
uᵢ = (i - p)/(n - p)    for i = p+1, ..., n
```

### Implementation

```cpp
// In clamped_knots() method (lines 95-104)
std::vector<double> bspline::clamped_knots(size_t cpCount, size_t degree) {
    size_t m = cpCount + degree;  // last index
    std::vector<double> U(m + 1);
    
    for (size_t i = 0; i <= m; ++i) {
        if (i <= degree)
            U[i] = 0.0;                                    // First p+1 knots
        else if (i >= cpCount)
            U[i] = 1.0;                                    // Last p+1 knots
        else
            U[i] = (double(i - degree)) / (cpCount - degree);  // Interior knots
    }
    return U;
}
```

### Interpolation Knot Vector (NURBS Book Algorithm A9.1)

For interpolating **m** data points with degree **p**, interior knots are averaged:

```
u_{j+p} = (1/p) · Σ(i=j to j+p-1) ūᵢ    for j = 1, ..., m-p-1
```

Where **ūᵢ** are the parameter values assigned to data points.

### Implementation

```cpp
// In interpolation_knots() helper (lines 166-179)
static std::vector<double> interpolation_knots(const std::vector<double> &u, size_t degree) {
    size_t m = u.size();
    size_t p = degree;
    size_t knotSize = m + p + 1;
    std::vector<double> U(knotSize);
    
    for (size_t i = 0; i <= p; ++i) U[i] = 0.0;
    for (size_t i = 0; i <= p; ++i) U[knotSize - 1 - i] = 1.0;
    
    if (m > p + 1) {
        for (size_t j = 1; j < m - p; ++j) {
            double s = 0.0;
            for (size_t i = j; i < j + p; ++i) s += u[i];
            U[j + p] = s / double(p);  // Average of p consecutive parameters
        }
    }
    return U;
}
```

---

## Standard Interpolation

Given **m** data points **Dᵢ** (i = 0, ..., m-1), find control points **Pⱼ** such that:

```
C(ūᵢ) = Dᵢ    for i = 0, ..., m-1
```

This leads to a linear system:

```
A · P = D
```

Where:

- **A** is an m×m matrix with **Aᵢⱼ = Nⱼ,ₚ(ūᵢ)**
- **P** is the m×d matrix of control points
- **D** is the m×d matrix of data points

### Solution

```
P = A⁻¹ · D
```

### Implementation

```cpp
// In interpolate() method (lines 181-220)
std::unique_ptr<bspline> bspline::interpolate(...) {
    // Build basis matrix A (m x m)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, m);
    
    for (size_t i = 0; i < m; ++i) {
        double ui = u[i];
        size_t span = helper.find_span(ui);
        auto N = helper.basis_function(ui);  // Get basis values
        size_t firstCol = span - degree;
        
        for (size_t r = 0; r < N.size(); ++r) {
            A(i, firstCol + r) = N[r];  // Aᵢⱼ = Nⱼ,ₚ(ūᵢ)
        }
    }
    
    // Build data matrix B (m x dim)
    Eigen::MatrixXd B(m, dim);
    for (size_t i = 0; i < m; ++i)
        B.row(i) = data_points[i].transpose();
    
    // Solve: P = A⁻¹ · B
    Eigen::MatrixXd P = A.fullPivLu().solve(B);
    
    return std::make_unique<bspline>(cps, degree, knots);
}
```

---

## Smooth Interpolation (Penalized Fitting)

For noisy data, we use **penalized least squares** to balance fitting and smoothness:

### Objective Function

```
minimize: ||A·P - D||² + λ·||L·P||²
```

Where:

- **||A·P - D||²** is the data fidelity term (fitting error)
- **||L·P||²** is the roughness penalty (second derivative)
- **λ ≥ 0** is the smoothing parameter

### Roughness Penalty Matrix

The second-difference operator **D₂** approximates the second derivative:

```
D₂ = [ 1  -2   1   0   0  ...  0 ]
     [ 0   1  -2   1   0  ...  0 ]
     [ 0   0   1  -2   1  ...  0 ]
     [          ...              ]
     [ 0  ...  0   1  -2   1   0 ]
```

Dimensions: **(n-2) × n** where **n** is the number of control points.

The roughness penalty matrix is:

```
R = D₂ᵀ · D₂
```

Dimensions: **n × n**

### Normal Equations

Taking the derivative and setting to zero:

```
∂/∂P [||A·P - D||² + λ·||L·P||²] = 0
```

This gives:

```
2·Aᵀ·A·P - 2·Aᵀ·D + 2λ·Lᵀ·L·P = 0
```

Simplifying:

```
(Aᵀ·A + λ·R)·P = Aᵀ·D
```

Where **R = Lᵀ·L = D₂ᵀ·D₂**.

### Solution

```
P = (Aᵀ·A + λ·R)⁻¹ · Aᵀ·D
```

### Implementation

```cpp
// In smooth_interpolate() method (lines 222-294)
std::unique_ptr<bspline> bspline::smooth_interpolate(...) {
    // Reduce number of control points for smoothing
    size_t cpCount = std::max<size_t>(degree + 1, std::min(m, (m + degree) / 2));
    
    // Build basis matrix A (m x cpCount)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, cpCount);
    // ... fill A with basis function values ...
    
    // Build data matrix B
    Eigen::MatrixXd B(m, dim);
    for (size_t i = 0; i < m; ++i)
        B.row(i) = data_points[i].transpose();
    
    // Build second-difference penalty matrix
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(cpCount, cpCount);
    if (cpCount > 3) {
        Eigen::MatrixXd D2 = Eigen::MatrixXd::Zero(cpCount - 2, cpCount);
        
        for (int i = 0; i < static_cast<int>(cpCount) - 2; ++i) {
            D2(i, i)     =  1.0;   // Second difference operator
            D2(i, i + 1) = -2.0;
            D2(i, i + 2) =  1.0;
        }
        
        R = D2.transpose() * D2;  // R = D₂ᵀ · D₂
    }
    
    // Form normal equations: (AᵀA + λR)·P = Aᵀ·B
    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::MatrixXd M = ATA + lambda * R;      // Left-hand side
    Eigen::MatrixXd RHS = A.transpose() * B;   // Right-hand side
    
    // Solve using Cholesky decomposition (LDLT)
    Eigen::MatrixXd P = M.ldlt().solve(RHS);
    
    return std::make_unique<bspline>(cps, degree, knots);
}
```

### Smoothing Parameter λ

- **λ = 0**: No smoothing (exact interpolation)
- **λ → ∞**: Maximum smoothing (approaches polynomial fit)
- **Typical values**: 0.01 to 10.0

**Effect on control points:**

- Small λ: Control points close to data points
- Large λ: Control points arranged smoothly, may deviate from data

---

## Parameterization Methods

Assigns parameter values **ūᵢ ∈ [0,1]** to data points **Dᵢ**.

### 1. Uniform Parameterization

```
ūᵢ = i/(m-1)    for i = 0, ..., m-1
```

Simple but may not reflect data point spacing.

### 2. Chord Length Parameterization (Default)

```
ū₀ = 0
ūᵢ = ūᵢ₋₁ + ||Dᵢ - Dᵢ₋₁|| / L    for i = 1, ..., m-2
ū_{m-1} = 1
```

Where **L = Σ||Dᵢ - Dᵢ₋₁||** is the total chord length.

Reflects the geometric spacing of data points.

### 3. Centripetal Parameterization

```
ūᵢ = ūᵢ₋₁ + √(||Dᵢ - Dᵢ₋₁||) / L'
```

Where **L' = Σ√(||Dᵢ - Dᵢ₋₁||)**.

Better for data with sharp corners or cusps.

### Implementation

```cpp
// In parameterize() helper (lines 139-159)
static std::vector<double> parameterize(const std::vector<Eigen::VectorXd> &pts,
                                        const std::string &method) {
    size_t m = pts.size();
    std::vector<double> u(m, 0.0);
    
    if (method == "uniform") {
        for (size_t i = 0; i < m; ++i)
            u[i] = (m == 1) ? 0.0 : double(i) / double(m - 1);
        return u;
    }
    
    // Chord length (default)
    double total = 0.0;
    for (size_t i = 1; i < m; ++i)
        total += (pts[i] - pts[i - 1]).norm();  // Accumulate chord lengths
    
    if (total == 0.0) {  // Degenerate case
        // Fall back to uniform
        for (size_t i = 0; i < m; ++i)
            u[i] = (m == 1) ? 0.0 : double(i) / double(m - 1);
        return u;
    }
    
    double acc = 0.0;
    for (size_t i = 1; i < m - 1; ++i) {
        acc += (pts[i] - pts[i - 1]).norm();
        u[i] = acc / total;  // Normalize to [0,1]
    }
    
    u.back() = 1.0;
    u.front() = 0.0;
    return u;
}
```

---

## Summary of Key Formulas

| Concept              | Formula                                                                         |
|----------------------|---------------------------------------------------------------------------------|
| **B-spline curve**   | C(u) = Σᵢ Nᵢ,ₚ(u)·Pᵢ                                                            |
| **Basis recursion**  | Nᵢ,ₚ(u) = [(u-uᵢ)/(uᵢ₊ₚ-uᵢ)]·Nᵢ,ₚ₋₁(u) + [(uᵢ₊ₚ₊₁-u)/(uᵢ₊ₚ₊₁-uᵢ₊₁)]·Nᵢ₊₁,ₚ₋₁(u) |
| **Interior knots**   | uᵢ = (i-p)/(n-p)                                                                |
| **Interpolation**    | A·P = D, where Aᵢⱼ = Nⱼ,ₚ(ūᵢ)                                                   |
| **Smooth fitting**   | minimize: ‖A·P-D‖² + λ·‖D₂·P‖²                                                  |
| **Normal equations** | (AᵀA + λR)·P = AᵀD, where R = D₂ᵀD₂                                             |
| **Chord length**     | ūᵢ = Σⱼ₌₁ⁱ ‖Dⱼ-Dⱼ₋₁‖ / L                                                        |

---

## References

1. **Piegl, L., & Tiller, W. (1997).** "The NURBS Book" (2nd ed.). Springer.
    - Algorithm A2.2: FindSpan (binary search for knot span)
    - Algorithm A2.3: BasisFuns (Cox-de Boor recursion)
    - Algorithm A3.5: Curve evaluation (de Boor's algorithm)
    - Algorithm A9.1: Global curve interpolation

2. **de Boor, C. (2001).** "A Practical Guide to Splines". Springer.
    - Cox-de Boor recursion formula
    - B-spline properties and theory

3. **Reinsch, C. H. (1967).** "Smoothing by spline functions." Numerische Mathematik.
    - Penalized least squares smoothing

---

**Author:** Francisco Nunez  
**Date:** November 30, 2025  
**File:** libs/interpolation/src/bspline.cpp

