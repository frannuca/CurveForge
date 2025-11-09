//
// Created by Francisco Nunez on 01.11.2025.
//

#include "../include/interpolation/linear.h"

double LinearInterpolation::lininterp(const std::vector<double> &x, const std::vector<double> &y, double xq) {
    if (xq <= x.front()) return y.front();
    if (xq >= x.back()) return y.back();
    auto it = upper_bound(x.begin(), x.end(), xq);
    size_t j = size_t(it - x.begin()), i = j - 1;
    double w = (xq - x[i]) / (x[j] - x[i]);
    return y[i] * (1.0 - w) + y[j] * w;
}
