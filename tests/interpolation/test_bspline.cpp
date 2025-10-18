//
// Created by Francisco Nunez on 27.09.2025.
//

#include "interpolation/bspline.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

int main() {
    using namespace interpolation;
    // Data points to interpolate (2D curve y = x^2 scaled domain [0,1])
    std::vector<Eigen::VectorXd> data;
    int N = 9;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d pt;
        double x = double(i) / (N - 1);
        pt << x, x * x;
        data.push_back(pt);
    }
    auto spline = bspline::interpolate(data, 3, "uniform");

    // Evaluate at original parameterization points and check closeness
    for (int i = 0; i < N; ++i) {
        double u = double(i) / (N - 1);
        auto val = spline->evaluate(u);
        double expectedY = u * u;
        assert(std::fabs(val[0]-u) < 1e-6); // x coordinate preserved
        assert(std::fabs(val[1]-expectedY) < 1e-3); // allow small interpolation tolerance
    }
    std::cout << "KNOTS_OK\n";
    return 0;
}
