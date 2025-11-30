//
// Created by Francisco Nunez on 30.11.2025.
//

#include "interpolation/BilinearInterpolation.h"

interpolation::BilinearInterpolation::BilinearInterpolation(std::vector<double> &&x, std::vector<double> &&y,
                                                            Eigen::MatrixXd &&z) : x_(std::move(x)), y_(std::move(y)),
    z_(std::move(z)) {
}

interpolation::BilinearInterpolation::BilinearInterpolation(const std::vector<double> &x, const std::vector<double> &y,
                                                            const Eigen::MatrixXd &z) : x_(x), y_(y), z_(z) {
}

double interpolation::BilinearInterpolation::interpolate(const double &x, const double &y) const {
    auto x_it = std::lower_bound(x_.begin(), x_.end(), x);
    auto y_it = std::lower_bound(y_.begin(), y_.end(), y);

    // Handle boundary cases
    if (x_it == x_.begin()) x_it++;
    if (y_it == y_.begin()) y_it++;
    if (x_it == x_.end()) x_it = x_.end() - 1;
    if (y_it == y_.end()) y_it = y_.end() - 1;

    size_t i1 = std::distance(x_.begin(), x_it) - 1;
    size_t i2 = i1 + 1;
    size_t j1 = std::distance(y_.begin(), y_it) - 1;
    size_t j2 = j1 + 1;

    if (i2 >= x_.size()) i2 = x_.size() - 1;
    if (j2 >= y_.size()) j2 = y_.size() - 1;

    double x1 = x_[i1];
    double x2 = x_[i2];
    double y1 = y_[j1];
    double y2 = y_[j2];

    if (x2 == x1 || y2 == y1) {
        // Return nearest value
        return z_(i1, j1);
    }

    // Bilinear interpolation
    double tx = (x - x1) / (x2 - x1);
    double ty = (y - y1) / (y2 - y1);

    double v11 = z_(i1, j1);
    double v12 = z_(i1, j2);
    double v21 = z_(i2, j1);
    double v22 = z_(i2, j2);

    return (1 - tx) * (1 - ty) * v11
           + (1 - tx) * ty * v12
           + tx * (1 - ty) * v21
           + tx * ty * v22;
}
